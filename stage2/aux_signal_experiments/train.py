"""Train one complementary-signal arm (LC-v2 recipe as in phase_study: AdamW 1e-3 wd .05, OneCycle, batch 4, 30 epochs,
patience 7, batch-1 eval, direct-decoder checkpoint selection). Features are read lazily from the shared dense cache.

--motion none | global (current M_motion 28-d input) | residual (residual_motion_features) | both (concatenated)
Motion inputs are per sampled position: max and mean of the per-native-frame descriptors over (previous sample, this sample].
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from stage2.long_context_v2_experiments import common as C
from stage2.long_context_v2_experiments.train import to_device
from stage2.phase_study.train import lazy_item, materialize
from stage2.spotting_experiments.objective import constrained_anchors
from stage2.spotting_experiments.train import normalized_metrics
from .losses import total_loss
from .model import AuxPyramid

HERE = C.REPO / "stage2/aux_signal_experiments"
RESULTS = HERE / "results"
RESIDUAL = HERE / "cache_residual"
GEO = HERE / "cache_geo"
HR = HERE / "cache_hr"
LANE = HERE / "cache_lane_v2"  # occupancy variant; the contact variant (cache_lane) failed QA (AUC at ENTRY 0.49)


def segments(per_frame, abs_idx):
    """max and mean over native frames (previous sample, this sample]; frame indices only."""
    out = np.zeros((len(abs_idx), 2 * per_frame.shape[1]), np.float32); prev = abs_idx[0] - 1
    for t, a in enumerate(abs_idx):
        seg = per_frame[max(min(prev + 1, a), 0):a + 1]; out[t] = np.concatenate([seg.max(0), seg.mean(0)]); prev = a
    return out


def residual_per_frame(sample_id):
    return np.load(RESIDUAL / f"{sample_id}.residual.npy").astype(np.float32)  # already squashed by extract_residual


def attach_inputs(it, motion, lane, geo=False):
    if geo:  # geometry-head features at the sampled frames, appended to the per-position input (see extract_geo.py)
        d = np.load(GEO / f"{it['sample_id']}.geo.npz"); pos = np.searchsorted(d["frames"], it["frame_numbers"].numpy())
        assert (d["frames"][pos] == it["frame_numbers"].numpy()).all()
        g = torch.from_numpy(d["feats"][pos].astype(np.float32))
        base = it["motion"] if motion in ("global", "residual", "both") else None
        if motion == "residual": base = torch.from_numpy(segments(residual_per_frame(it["sample_id"]), it["abs_idx"]))
        elif motion == "both": base = torch.cat([it["motion"], torch.from_numpy(segments(residual_per_frame(it["sample_id"]), it["abs_idx"]))], 1)
        it["motion"] = g if base is None else torch.cat([base, g], 1)
        motion = "_done"
    if motion == "residual": it["motion"] = torch.from_numpy(segments(residual_per_frame(it["sample_id"]), it["abs_idx"]))
    elif motion == "both":
        it["motion"] = torch.cat([it["motion"], torch.from_numpy(segments(residual_per_frame(it["sample_id"]), it["abs_idx"]))], 1)
    if lane != "none":
        d = np.load(LANE / f"{it['sample_id']}.lane.npz")
        pos = np.searchsorted(d["frames"], it["frame_numbers"].numpy()); assert (d["frames"][pos] == it["frame_numbers"].numpy()).all()
        it["lane_state"] = torch.from_numpy(d["state"][pos].astype(np.int64)); it["lane_score"] = torch.from_numpy(d["score"][pos].astype(np.float32))
        # L3 target: the last OUTSIDE/APPROACHING -> CROSSING+ transition at or before COLLISION (training target only)
        st = it["lane_state"].numpy(); c = it["collision_index"]
        up = [t for t in range(1, min(c, len(st) - 1) + 1) if 0 <= st[t - 1] <= 1 and st[t] >= 2]
        it["lane_onset"] = int(up[-1]) if up and d["valid_clip"] else -1
    return it


def extra_items(train_split, entry_w, name=None):
    """Pseudo/metadata-labelled unlabelled NEXAR clips for this split (labels_<fold k | fixed>.json), 'both' motion input."""
    import json, re
    from .nexar_labels import unl_item, OUT as UOUT
    if name is None:
        m = re.search(r"fold(\d)_train", str(train_split)); name = f"fold{m.group(1)}" if m else "fixed"
    labels = json.loads((UOUT / f"labels_{name}.json").read_text()); items = []
    for sid, lab in labels.items():
        fr, _, mot = unl_item(sid); f = torch.from_numpy(fr.copy())
        norm = (fr - fr[0]).astype(np.float32) / max(int(fr[-1] - fr[0]), 1)
        items.append({"sample_id": sid, "source_id": f"NEXARU:{sid}", "frame_numbers": f, "normalized_positions": torch.from_numpy(norm),
                      "entry_index": int(np.abs(fr - lab["entry_frame"]).argmin()), "collision_index": int(np.abs(fr - lab["collision_frame"]).argmin()),
                      "entry_frame": lab["entry_frame"], "collision_frame": lab["collision_frame"], "entry_side": 0, "evasion": 0,
                      "num_available_frames": lab["nframes"], "abs_idx": fr, "x_path": str(UOUT / f"{sid}.npy"), "x_shape": len(fr),
                      "motion": mot, "attr_w": 0.0, "entry_w": entry_w})
    return items, name


STRIDE = C.REPO / "stage2/generalization/cache_stride"
OBJ = C.REPO / "stage2/generalization/cache_objmotion"


def add_obj(it, k=1):
    """append object-level independent motion (extract_objmotion.py, stride-k cache) segments to the motion input"""
    o = np.load(OBJ / f"k{k}" / f"{it['sample_id']}.npy")
    pos = it["_pos"] if "_pos" in it else it["abs_idx"]
    it["motion"] = torch.cat([it["motion"], torch.from_numpy(segments(o, pos))], 1)
    return it


def stride_item(base, k):
    """The same labelled clip seen at 1/k of its frame rate: native frames[::k], adaptive re-sampling of the reduced sequence,
    DINO tokens of the retained frames (lazy, via abs_idx), motion recomputed on retained frames (extract_stride_motion.py)."""
    from .nexar_labels import motion28
    sid = base["sample_id"]; frames_all, _ = C.dense(sid); d = np.load(STRIDE / f"k{k}" / f"{sid}.npz")
    kept = d["kept"]; reduced = frames_all[kept]; pos = C.select_adaptive(reduced); fr = reduced[pos]
    g = segments(motion28(d["motion"]), pos); r = segments(d["residual"], pos)
    norm = (fr - fr[0]).astype(np.float32) / max(int(fr[-1] - fr[0]), 1)
    it = {k_: v for k_, v in base.items() if k_ not in ("motion", "abs_idx", "frame_numbers", "normalized_positions", "x_shape")}
    it.update(frame_numbers=torch.from_numpy(fr.copy()), normalized_positions=torch.from_numpy(norm), abs_idx=kept[pos], x_shape=len(fr),
              entry_index=int(np.abs(fr - base["entry_frame"]).argmin()), collision_index=int(np.abs(fr - base["collision_frame"]).argmin()),
              num_available_frames=len(reduced), motion=torch.from_numpy(np.concatenate([g, r], 1)), _pos=pos)
    it["phase_entry_index"], it["phase_collision_index"] = it["entry_index"], it["collision_index"]
    return it


def materialize_any(it):
    out = {**it, "x": torch.from_numpy(np.ascontiguousarray(np.load(it["x_path"], mmap_mode="r")))} if "x_path" in it else materialize(it)
    if it.get("hr_path"): out["hr"] = torch.from_numpy(np.ascontiguousarray(np.load(it["hr_path"], mmap_mode="r")))
    return out


def collate(items, motion, lane):
    if motion not in ("none", "global"):  # collate_m assumes the 28-d global motion input
        b = C.collate_m([{**it, "motion": None} for it in items])
    else:
        b = C.collate_m(items)
    if motion not in ("none", "global"):
        m = torch.zeros(len(items), b["x"].shape[1], items[0]["motion"].shape[1])
        for j, it in enumerate(items): m[j, :len(it["motion"])] = it["motion"]
        b["motion"] = m
    if lane != "none":
        T = b["x"].shape[1]
        b["lane_state"] = torch.full((len(items), T), -1, dtype=torch.long); b["lane_score"] = torch.full((len(items), T), -1.0)
        for j, it in enumerate(items):
            n = len(it["lane_state"]); b["lane_state"][j, :n] = it["lane_state"]; b["lane_score"][j, :n] = it["lane_score"]
        b["lane_onset"] = torch.tensor([it["lane_onset"] for it in items])
    if any("hr" in it for it in items):
        hr = torch.zeros(len(items), b["x"].shape[1], 350, 384, dtype=torch.float16)
        for j, it in enumerate(items): hr[j, :len(it["hr"])] = it["hr"]
        b["hr"] = hr
    if any("attr_w" in it for it in items):
        b["attr_w"] = torch.tensor([it.get("attr_w", 1.0) for it in items]); b["entry_w"] = torch.tensor([it.get("entry_w", 1.0) for it in items])
    return b


@torch.inference_mode()
def infer(model, items, device, motion, lane):
    model.eval(); preds = []
    for it in items:
        it = materialize_any(it)
        b = collate([it], motion, lane); x, v = b["x"].to(device), b["time_valid"].to(device)
        hr = b["hr"].to(device) if "hr" in b else None
        o = model(x, v, motion=b["motion"].to(device), hr=hr) if model.uses_motion else model(x, v)
        ei, ci = constrained_anchors(o["entry_logits"], o["collision_logits"])
        frames = it["frame_numbers"].numpy()
        p = {"sample_id": it["sample_id"], "source_id": it["source_id"],
             "entry_frame": int(frames[int(ei[0])]), "collision_frame": int(frames[int(ci[0])]),
             "entry_side": int(o["side_logits"][0].argmax()), "evasion_space": int(o["evasion_logits"][0] >= 0),
             "entry_gt": it["entry_frame"], "collision_gt": it["collision_frame"],
             "entry_side_gt": it["entry_side"], "evasion_gt": it["evasion"],
             "entry_index_gt": it["entry_index"], "collision_index_gt": it["collision_index"],
             "num_available_frames": it["num_available_frames"], "num_positions": len(frames), "frames": frames.tolist(),
             "entry_logits": o["entry_logits"][0].float().cpu().tolist(), "collision_logits": o["collision_logits"][0].float().cpu().tolist(),
             "side_prob_right": float(o["side_logits"][0].float().softmax(-1)[1]), "evasion_prob": float(o["evasion_logits"][0].float().sigmoid())}
        for k in ("phase_logp", "risk_logits", "bnd_entry_logits", "bnd_collision_logits", "lane_logits"):
            if k in o: p[k] = o[k][0].float().cpu().numpy().round(4).tolist()
        preds.append(p)
    return preds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--base-loss", choices=["nt", "p2"], default="nt")
    p.add_argument("--phase-rep", choices=["none", "cat", "ord"], default="none")
    p.add_argument("--w-direct", type=float, default=1.0); p.add_argument("--w-phase", type=float, default=0.0)
    p.add_argument("--w-tr", type=float, default=0.0); p.add_argument("--w-mono", type=float, default=0.0)
    p.add_argument("--risk", choices=["none", "R1", "R2", "R3"], default="none"); p.add_argument("--w-risk", type=float, default=0.0)
    p.add_argument("--boundary", choices=["none", "bnd1", "bnd2", "bnd3"], default="none")
    p.add_argument("--bnd-width", type=float, default=1.0); p.add_argument("--w-bnd", type=float, default=0.0)
    p.add_argument("--lane", choices=["none", "cat4", "ord"], default="none"); p.add_argument("--w-lane", type=float, default=0.0)
    p.add_argument("--w-lane-tr", type=float, default=0.0)
    p.add_argument("--motion", choices=["none", "global", "residual", "both"], default="none")
    p.add_argument("--geo", action="store_true", help="append geometry-head features (cache_geo) to the per-position input")
    p.add_argument("--mask-entry-sources", default="", help="comma list of sources whose ENTRY loss is masked (e.g. MMAU; label-consistency test)")
    p.add_argument("--objmotion", action="store_true", help="append object-level independent motion (requires --motion both)")
    p.add_argument("--stride-aug", default="", help="temporal-rate augmentation, e.g. '0.5,0.25,0.25' = P(stride 1,2,3) per clip per epoch")
    p.add_argument("--clip-norm", choices=["none", "x", "m", "xm"], default="none", help="per-clip input normalisation (see model.py)")
    p.add_argument("--balance-sources", action="store_true", help="sample training clips with probability 1/(source count)")
    p.add_argument("--hr", action="store_true", help="add 14x25 high-resolution frozen tokens (cache_hr) to the input")
    p.add_argument("--ema", type=float, default=0.0, help="EMA decay of weights used for evaluation/selection (X_ema family: 0.99)")
    p.add_argument("--w-entry", type=float, default=1.0, help="ENTRY weight in the NT direct loss (1 = NT)")
    p.add_argument("--sigma", type=float, default=1.0, help="direct-target Gaussian width in sampled positions (1 = NT)")
    p.add_argument("--extra-nexar", action="store_true", help="add the metadata/pseudo-labelled unlabelled NEXAR clips (nexar_labels.py)")
    p.add_argument("--extra-labels", default=None, help="label set name for --extra-nexar (default: from --train-split: foldK / fixed)")
    p.add_argument("--stop-epoch", type=int, default=0, help="full-data refit: no selection, save the weights after this epoch")
    p.add_argument("--extra-entry-w", type=float, default=0.5, help="loss weight of pseudo-labelled ENTRY on the extra clips")
    p.add_argument("--epochs", type=int, default=30); p.add_argument("--patience", type=int, default=7)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--train-split", default="train"); p.add_argument("--val-split", default="val"); p.add_argument("--output", default=None)
    a = p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    torch.set_num_threads(2); device = torch.device("cuda")
    out = Path(a.output) if a.output else RESULTS / a.run_id / f"seed{a.seed}"
    if (out / "checkpoint.pt").exists(): raise SystemExit(f"{out} exists; refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    train_rows, val_rows = C.rows(a.train_split), C.rows(a.val_split)
    train_items = [attach_inputs(lazy_item(r), a.motion, a.lane, a.geo) for r in train_rows]
    val_items = [attach_inputs(lazy_item(r), a.motion, a.lane, a.geo) for r in val_rows]
    mode = "custom" if (a.geo or a.motion in ("residual", "both")) else a.motion  # collate: custom-width input tensor
    extra, extra_name = (extra_items(a.train_split, a.extra_entry_w, a.extra_labels) if a.extra_nexar else ([], None))
    if extra: assert a.motion == "both" and not a.geo and a.lane == "none", "extra NEXAR clips support the 'both' motion input only"
    if a.mask_entry_sources:  # keep these clips for COLLISION + attributes, drop their ENTRY supervision
        masked = set(a.mask_entry_sources.split(","))
        for it in train_items: it["attr_w"] = 1.0; it["entry_w"] = 0.0 if C.source(it) in masked else 1.0
        assert a.base_loss == "nt", "--mask-entry-sources uses the weighted NT loss"
    if a.objmotion:
        assert a.motion == "both" and not a.geo and not extra, "--objmotion is appended to the 'both' motion input"
        for it in train_items + val_items: add_obj(it, 1)
    views = {}
    if a.stride_aug:
        assert a.motion == "both" and not a.geo and not a.hr and a.lane == "none", "stride augmentation supports the 'both' motion input only"
        probs = [float(x) for x in a.stride_aug.split(",")]
        views = {k: [stride_item(it, k) for it in train_items] for k in range(2, len(probs) + 1) if probs[k - 1] > 0}
        if a.objmotion: views = {k: [add_obj(it, k) for it in v] for k, v in views.items()}
    for it in train_items + extra: it["phase_entry_index"], it["phase_collision_index"] = it["entry_index"], it["collision_index"]
    if a.hr:
        assert not extra, "HR tokens are not extracted for the extra NEXAR clips"
        assert a.motion == "both", "--hr is combined with the residual-motion input (E4-style) only"
        for it in train_items + val_items:
            it["hr_path"] = str(HR / f"{it['sample_id']}.npy")
            assert (np.load(HR / f"{it['sample_id']}.frames.npy") == it["frame_numbers"].numpy()).all()
    motion_dim = 0 if (a.motion == "none" and not a.geo) else int(train_items[0]["motion"].shape[1])
    if a.objmotion: mode = "custom"
    if a.hr and motion_dim == 0: mode = "custom"
    cfg = dict(vars(a), hr=a.hr, extra_labels=extra_name, n_extra=len(extra), soft_width=0.0, lr=1e-3, weight_decay=.05, strict_fps_blind=True, selection="direct",
               motion_dim=motion_dim, n_train=len(train_rows), n_val=len(val_rows))
    model = AuxPyramid(phase_rep=a.phase_rep, risk=a.risk != "none", boundary=a.boundary, lane=a.lane, motion_dim=motion_dim, hr=a.hr, clip_norm=a.clip_norm).to(device)
    params = sum(x.numel() for x in model.parameters())
    import copy
    ema = copy.deepcopy(model).eval() if a.ema else None
    if ema is not None:
        for x in ema.parameters(): x.requires_grad_(False)
    m_eval = ema if ema is not None else model
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.05)
    n_extra = min(len(extra), len(train_rows))  # per epoch: all labelled clips + an equal-size random draw of extra clips
    steps = (len(train_rows) + n_extra + a.batch_size - 1) // a.batch_size
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=a.epochs * steps, pct_start=.1)
    (out / "config.json").write_text(json.dumps({**cfg, "parameter_count": params}, indent=2) + "\n")
    gen = torch.Generator().manual_seed(a.seed)

    def make_batch(idx):
        items = [materialize_any(pool[i]) for i in idx]
        b = collate(items, mode, a.lane)
        b["phase_entry_index"], b["phase_collision_index"] = b["entry_index"], b["collision_index"]
        return to_device(b, device)

    history, best, best_epoch, stale, best_state = [], -1.0, 0, 0, None; wall = time.perf_counter()
    for epoch in range(1, a.epochs + 1):
        model.train(); sums = {}
        pick = torch.randperm(len(extra), generator=gen)[:n_extra].tolist() if extra else []
        if views:  # temporal-rate augmentation: each labelled clip at stride 1/2/3 with the given probabilities
            ks = torch.multinomial(torch.tensor(probs), len(train_items), replacement=True, generator=gen).tolist()
            base = [train_items[i] if ks[i] == 0 else views[ks[i] + 1][i] for i in range(len(train_items))]
        else:
            base = train_items
        pool = base + [extra[i] for i in pick]
        if a.balance_sources:  # same number of samples per epoch, drawn with probability 1 / (clips of that source)
            from collections import Counter
            srcs = [C.source(it) for it in pool]; cnt = Counter(srcs)
            w = torch.tensor([1.0 / cnt[s_] for s_ in srcs])
            order = torch.multinomial(w, len(pool), replacement=True, generator=gen).tolist()
        else:
            order = torch.randperm(len(pool), generator=gen).tolist()
        for s in range(0, len(order), a.batch_size):
            batch = make_batch(order[s:s + a.batch_size])
            o = model(batch["x"], batch["time_valid"], motion=batch["motion"], hr=batch.get("hr")) if model.uses_motion else model(batch["x"], batch["time_valid"])
            loss, parts = total_loss(o, batch, cfg)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            if ema is not None:
                with torch.no_grad():
                    for t_, s_ in zip(ema.parameters(), model.parameters()): t_.lerp_(s_, 1 - a.ema)
                    for t_, s_ in zip(ema.buffers(), model.buffers()): t_.copy_(s_)
            for k, v in {"loss": loss, **parts}.items(): sums[k] = sums.get(k, 0.0) + float(v)
        if a.stop_epoch and epoch != a.stop_epoch:  # fixed-epoch mode: per-epoch validation is logging only -> skip it
            history.append({"epoch": epoch, **{f"train_{k}": v / steps for k, v in sums.items()}}); continue
        val_pred = infer(m_eval, val_items, device, mode, a.lane)
        score = normalized_metrics(val_pred)["fpsblind_selection_score"]
        rec = {"epoch": epoch, **{f"train_{k}": v / steps for k, v in sums.items()}, "val_fpsblind_score": score, "val_official": C.metrics(val_pred)["score"]}
        history.append(rec); print(json.dumps(rec), flush=True)
        if a.stop_epoch:  # full-data refit: fixed epoch, no validation-based selection
            if epoch == a.stop_epoch:
                best, best_epoch = score, epoch; best_state = {k: v.detach().cpu().clone() for k, v in m_eval.state_dict().items()}; break
            continue
        if score > best + 1e-8:
            best, best_epoch, stale = score, epoch, 0; best_state = {k: v.detach().cpu().clone() for k, v in m_eval.state_dict().items()}
        else: stale += 1
        if stale >= a.patience: break
    m_eval.load_state_dict(best_state)
    t0 = time.perf_counter(); preds = infer(m_eval, val_items, device, mode, a.lane); head_ms = 1000 * (time.perf_counter() - t0) / len(preds)
    train_pred = infer(m_eval, train_items, device, mode, a.lane) if not a.stop_epoch else preds
    summary = {"breakdown": C.breakdown(preds), "best_epoch": best_epoch, "last_epoch": epoch,
               "train_official_at_best": C.metrics(train_pred)["score"], "val_fpsblind_at_best": best,
               "runtime": {"wall_seconds": time.perf_counter() - wall, "parameter_count": params, "head_ms_per_video": head_ms}}
    torch.save({"model": best_state, "config": cfg}, out / "checkpoint.pt")
    C.dump(out / "history.json", history); C.dump(out / "metrics.json", summary); C.dump(out / "predictions.json", preds)
    print("RESULT", a.run_id, a.seed, out.name, C.short_table(summary["breakdown"]), f"| best_ep {best_epoch}", flush=True)


if __name__ == "__main__": main()
