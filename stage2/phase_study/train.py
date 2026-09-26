"""Train one phase-study arm (LC-v2 recipe: AdamW 1e-3 wd .05, OneCycle, batch 4, 30 epochs, patience 7, batch-1 eval).

Checkpoint selection ALWAYS uses direct decoding (val fpsblind_selection_score), for every arm, so phase arms and their
no-phase controls are selected identically (the old PH_phase runs were selected with the structured decoder).
Saves predictions with direct-event logits, implied phase log-probs and side/evasion probabilities (for OOF ensembles
and offline decoders).
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
from stage2.spotting_experiments.objective import constrained_anchors
from stage2.spotting_experiments.train import normalized_metrics
from .losses import total_loss
from .model import ATTACH, PhasePyramid

RESULTS = C.REPO / "stage2/phase_study/results"


def lazy_item(row):
    """make_item without the feature copy: features are read from the shared memory-mapped dense cache at batch time
    (identical values; keeps each process at ~1 GB instead of ~4 GB)."""
    it = C.make_item(row, "adaptive")
    frames_all, _ = C.dense(row["sample_id"])
    it["abs_idx"] = np.searchsorted(frames_all, it["frame_numbers"].numpy())
    it["x_shape"] = len(it["x"]); del it["x"]
    return it


def materialize(it):
    _, feats = C.dense(it["sample_id"])
    return {**it, "x": torch.from_numpy(np.ascontiguousarray(feats[it["abs_idx"]]))}
SHARED = ("norm.", "token.", "frame.", "blocks.", "fusions.", "motion_proj.")


def shuffled_boundaries(items, seed, min_gap=0.10):
    """Donor phase boundaries: another training clip's normalized ENTRY/COLLISION positions mapped onto this clip's
    sampled positions. Donors must differ from the true normalized ENTRY and COLLISION by > min_gap (alignment destroyed);
    the class-frequency distribution is approximately preserved (boundaries come from real clips)."""
    rng = np.random.default_rng(10_000 + seed)
    frac = np.array([[it["entry_index"] / max(it["x_shape"] - 1, 1), it["collision_index"] / max(it["x_shape"] - 1, 1)] for it in items])
    out, dist = [], []
    for i, it in enumerate(items):
        ok = np.where((np.abs(frac[:, 0] - frac[i, 0]) > min_gap) & (np.abs(frac[:, 1] - frac[i, 1]) > min_gap))[0]
        j = int(rng.choice(ok if len(ok) else np.delete(np.arange(len(items)), i)))
        t = it["x_shape"]; e = int(round(frac[j, 0] * (t - 1))); c = int(round(frac[j, 1] * (t - 1)))
        e = min(e, t - 2); c = min(max(c, e + 1), t - 1)
        out.append((e, c)); dist.append(abs(frac[j] - frac[i]).mean())
    return out, float(np.mean(dist))


@torch.inference_mode()
def infer(model, items, device):
    model.eval(); preds = []
    for it in items:
        it = materialize(it) if "x" not in it else it
        b = C.collate_m([it]); x, v = b["x"].to(device), b["time_valid"].to(device)
        o = model(x, v, motion=b["motion"].to(device)) if model.uses_motion else model(x, v)
        ei, ci = constrained_anchors(o["entry_logits"], o["collision_logits"])
        frames = it["frame_numbers"].numpy(); t = len(frames)
        p = {"sample_id": it["sample_id"], "source_id": it["source_id"],
             "entry_frame": int(frames[int(ei[0])]), "collision_frame": int(frames[int(ci[0])]),
             "entry_side": int(o["side_logits"][0].argmax()), "evasion_space": int(o["evasion_logits"][0] >= 0),
             "entry_gt": it["entry_frame"], "collision_gt": it["collision_frame"],
             "entry_side_gt": it["entry_side"], "evasion_gt": it["evasion"],
             "entry_index_gt": it["entry_index"], "collision_index_gt": it["collision_index"],
             "num_available_frames": it["num_available_frames"], "num_positions": t, "frames": frames.tolist(),
             "entry_logits": o["entry_logits"][0].float().cpu().tolist(), "collision_logits": o["collision_logits"][0].float().cpu().tolist(),
             "side_prob_right": float(o["side_logits"][0].float().softmax(-1)[1]), "evasion_prob": float(o["evasion_logits"][0].float().sigmoid())}
        if "phase_logp" in o: p["phase_logp"] = o["phase_logp"][0].float().cpu().numpy().round(4).tolist()
        preds.append(p)
    return preds


def grad_cosines(model, batches, cfg, device):
    """Cosine similarity of per-loss gradients on the shared temporal-pyramid parameters (eval mode: no dropout noise)."""
    params = [p for n, p in model.named_parameters() if n.startswith(SHARED)]
    was = model.training; model.eval(); sims = {}
    for batch in batches:
        o = model(batch["x"], batch["time_valid"])
        _, parts = total_loss(o, batch, cfg)
        g = {}
        for k in ("direct", "phase", "transition"):
            if k in parts:
                gs = torch.autograd.grad(parts[k], params, retain_graph=True, allow_unused=True)
                g[k] = torch.cat([(x if x is not None else torch.zeros_like(p)).flatten() for x, p in zip(gs, params)])
        for a, b in (("direct", "phase"), ("direct", "transition"), ("phase", "transition")):
            if a in g and b in g and g[a].norm() > 0 and g[b].norm() > 0:
                sims.setdefault(f"{a}~{b}", []).append(float(torch.nn.functional.cosine_similarity(g[a], g[b], 0)))
        if "phase" in g: sims.setdefault("norm_ratio_phase/direct", []).append(float(g["phase"].norm() / g["direct"].norm()))
    model.train(was)
    return {k: float(np.mean(v)) for k, v in sims.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--phase-rep", choices=["none", "cat", "ord"], default="none")
    p.add_argument("--attach", choices=ATTACH, default="final")
    p.add_argument("--w-direct", type=float, default=1.0)
    p.add_argument("--w-phase", type=float, default=0.0)
    p.add_argument("--w-tr", type=float, default=0.0)
    p.add_argument("--w-mono", type=float, default=0.0)
    p.add_argument("--soft-width", type=float, default=0.0, help="phase-boundary softness in sampled positions (0 = hard)")
    p.add_argument("--phase-target", choices=["gt", "shuffled"], default="gt")
    p.add_argument("--grad-cos", type=int, default=0, help="log gradient cosines on N fixed train batches each epoch")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--train-split", default="train")
    p.add_argument("--val-split", default="val")
    p.add_argument("--output", default=None)
    a = p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    torch.set_num_threads(2)
    device = torch.device("cuda")
    out = Path(a.output) if a.output else RESULTS / a.run_id / f"seed{a.seed}"
    if (out / "checkpoint.pt").exists(): raise SystemExit(f"{out} exists; refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)

    train_rows = C.rows(a.train_split); val_rows = C.rows(a.val_split)
    train_items = [lazy_item(r) for r in train_rows]
    val_items = [lazy_item(r) for r in val_rows]
    shuffle_dist = None
    if a.phase_target == "shuffled":
        bounds, shuffle_dist = shuffled_boundaries(train_items, a.seed)
    else:
        bounds = [(it["entry_index"], it["collision_index"]) for it in train_items]
    for it, (e, c) in zip(train_items, bounds): it["phase_entry_index"], it["phase_collision_index"] = e, c

    cfg = dict(vars(a), recipe="LCv2/NT", lr=1e-3, weight_decay=.05, strict_fps_blind=True, selection="direct",
               n_train=len(train_rows), n_val=len(val_rows), shuffle_mean_abs_norm_shift=shuffle_dist)
    model = PhasePyramid(phase_rep=a.phase_rep, attach=a.attach).to(device)
    params = sum(x.numel() for x in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.05)
    steps = (len(train_rows) + a.batch_size - 1) // a.batch_size
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=a.epochs * steps, pct_start=.1)
    (out / "config.json").write_text(json.dumps({**cfg, "parameter_count": params}, indent=2) + "\n")
    gen = torch.Generator().manual_seed(a.seed)

    def make_batch(idx):
        items = [materialize(train_items[i]) for i in idx]
        b = C.collate_m(items)
        b["phase_entry_index"] = torch.tensor([it["phase_entry_index"] for it in items])
        b["phase_collision_index"] = torch.tensor([it["phase_collision_index"] for it in items])
        return to_device(b, device)

    cos_batches = [make_batch(list(range(i * a.batch_size, (i + 1) * a.batch_size))) for i in range(a.grad_cos)] \
        if a.grad_cos and a.phase_rep != "none" else []
    history, best, best_epoch, stale, best_state = [], -1.0, 0, 0, None
    wall = time.perf_counter()
    if cos_batches: history.append({"epoch": 0, "grad_cos": grad_cosines(model, cos_batches, cfg, device)})
    for epoch in range(1, a.epochs + 1):
        model.train(); sums = {}
        order = torch.randperm(len(train_items), generator=gen).tolist()
        for s in range(0, len(order), a.batch_size):
            batch = make_batch(order[s:s + a.batch_size])
            o = model(batch["x"], batch["time_valid"])
            loss, parts = total_loss(o, batch, cfg)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            for k, v in {"loss": loss, **parts}.items(): sums[k] = sums.get(k, 0.0) + float(v)
        val_pred = infer(model, val_items, device)
        score = normalized_metrics(val_pred)["fpsblind_selection_score"]
        rec = {"epoch": epoch, **{f"train_{k}": v / steps for k, v in sums.items()}, "val_fpsblind_score": score,
               "val_official": C.metrics(val_pred)["score"]}
        if cos_batches: rec["grad_cos"] = grad_cosines(model, cos_batches, cfg, device)
        history.append(rec); print(json.dumps(rec), flush=True)
        if score > best + 1e-8:
            best, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= a.patience: break
    model.load_state_dict(best_state)
    preds = infer(model, val_items, device)
    train_pred = infer(model, train_items, device)
    summary = {"breakdown": C.breakdown(preds), "best_epoch": best_epoch, "last_epoch": epoch,
               "train_fpsblind_at_best": normalized_metrics(train_pred)["fpsblind_selection_score"],
               "val_fpsblind_at_best": best, "train_official_at_best": C.metrics(train_pred)["score"],
               "runtime": {"wall_seconds": time.perf_counter() - wall, "parameter_count": params}}
    torch.save({"model": best_state, "config": cfg}, out / "checkpoint.pt")
    C.dump(out / "history.json", history); C.dump(out / "metrics.json", summary); C.dump(out / "predictions.json", preds)
    print("RESULT", a.run_id, a.seed, out.name, C.short_table(summary["breakdown"]), f"| best_ep {best_epoch}", flush=True)


if __name__ == "__main__": main()
