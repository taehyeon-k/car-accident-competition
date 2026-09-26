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
LANE = HERE / "cache_lane_v2"  # occupancy variant; the contact variant (cache_lane) failed QA (AUC at ENTRY 0.49)


def segments(per_frame, abs_idx):
    """max and mean over native frames (previous sample, this sample]; frame indices only."""
    out = np.zeros((len(abs_idx), 2 * per_frame.shape[1]), np.float32); prev = abs_idx[0] - 1
    for t, a in enumerate(abs_idx):
        seg = per_frame[max(min(prev + 1, a), 0):a + 1]; out[t] = np.concatenate([seg.max(0), seg.mean(0)]); prev = a
    return out


def residual_per_frame(sample_id):
    return np.load(RESIDUAL / f"{sample_id}.residual.npy").astype(np.float32)  # already squashed by extract_residual


def attach_inputs(it, motion, lane):
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
    return b


@torch.inference_mode()
def infer(model, items, device, motion, lane):
    model.eval(); preds = []
    for it in items:
        it = materialize(it)
        b = collate([it], motion, lane); x, v = b["x"].to(device), b["time_valid"].to(device)
        o = model(x, v, motion=b["motion"].to(device)) if model.uses_motion else model(x, v)
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
    train_items = [attach_inputs(lazy_item(r), a.motion, a.lane) for r in train_rows]
    val_items = [attach_inputs(lazy_item(r), a.motion, a.lane) for r in val_rows]
    for it in train_items: it["phase_entry_index"], it["phase_collision_index"] = it["entry_index"], it["collision_index"]
    motion_dim = 0 if a.motion == "none" else int(train_items[0]["motion"].shape[1])
    cfg = dict(vars(a), soft_width=0.0, lr=1e-3, weight_decay=.05, strict_fps_blind=True, selection="direct",
               motion_dim=motion_dim, n_train=len(train_rows), n_val=len(val_rows))
    model = AuxPyramid(phase_rep=a.phase_rep, risk=a.risk != "none", boundary=a.boundary, lane=a.lane, motion_dim=motion_dim).to(device)
    params = sum(x.numel() for x in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.05)
    steps = (len(train_rows) + a.batch_size - 1) // a.batch_size
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=a.epochs * steps, pct_start=.1)
    (out / "config.json").write_text(json.dumps({**cfg, "parameter_count": params}, indent=2) + "\n")
    gen = torch.Generator().manual_seed(a.seed)

    def make_batch(idx):
        items = [materialize(train_items[i]) for i in idx]
        b = collate(items, a.motion, a.lane)
        b["phase_entry_index"], b["phase_collision_index"] = b["entry_index"], b["collision_index"]
        return to_device(b, device)

    history, best, best_epoch, stale, best_state = [], -1.0, 0, 0, None; wall = time.perf_counter()
    for epoch in range(1, a.epochs + 1):
        model.train(); sums = {}
        order = torch.randperm(len(train_items), generator=gen).tolist()
        for s in range(0, len(order), a.batch_size):
            batch = make_batch(order[s:s + a.batch_size])
            o = model(batch["x"], batch["time_valid"], motion=batch["motion"]) if model.uses_motion else model(batch["x"], batch["time_valid"])
            loss, parts = total_loss(o, batch, cfg)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            for k, v in {"loss": loss, **parts}.items(): sums[k] = sums.get(k, 0.0) + float(v)
        val_pred = infer(model, val_items, device, a.motion, a.lane)
        score = normalized_metrics(val_pred)["fpsblind_selection_score"]
        rec = {"epoch": epoch, **{f"train_{k}": v / steps for k, v in sums.items()}, "val_fpsblind_score": score, "val_official": C.metrics(val_pred)["score"]}
        history.append(rec); print(json.dumps(rec), flush=True)
        if score > best + 1e-8:
            best, best_epoch, stale = score, epoch, 0; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else: stale += 1
        if stale >= a.patience: break
    model.load_state_dict(best_state)
    t0 = time.perf_counter(); preds = infer(model, val_items, device, a.motion, a.lane); head_ms = 1000 * (time.perf_counter() - t0) / len(preds)
    train_pred = infer(model, train_items, device, a.motion, a.lane)
    summary = {"breakdown": C.breakdown(preds), "best_epoch": best_epoch, "last_epoch": epoch,
               "train_official_at_best": C.metrics(train_pred)["score"], "val_fpsblind_at_best": best,
               "runtime": {"wall_seconds": time.perf_counter() - wall, "parameter_count": params, "head_ms_per_video": head_ms}}
    torch.save({"model": best_state, "config": cfg}, out / "checkpoint.pt")
    C.dump(out / "history.json", history); C.dump(out / "metrics.json", summary); C.dump(out / "predictions.json", preds)
    print("RESULT", a.run_id, a.seed, out.name, C.short_table(summary["breakdown"]), f"| best_ep {best_epoch}", flush=True)


if __name__ == "__main__": main()
