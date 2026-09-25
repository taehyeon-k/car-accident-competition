"""Train one LCPyramid variant on the fixed 279/70 split from the dense cache.

Recipe identical to temporal_pyramid_experiments.train P2 (AdamW lr 1e-3 wd .05, OneCycle, batch 4,
30 epochs, patience 7, soft_normalized targets width .015, selection on val fpsblind_selection_score),
plus the controlled options below.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from stage2.spotting_experiments.data import collate
from stage2.spotting_experiments.train import normalized_metrics
from stage2.temporal_pyramid_experiments.train import experiment_loss
from . import common as C
from .diagnostics import recall_at_k, summarize_recall
from .models import LCPyramid

RESULTS = C.REPO / "stage2/long_context_v2_experiments/results"


def phase_targets(batch):
    f = batch["frame_numbers"]
    e = batch["entry_frame"][:, None]; c = batch["collision_frame"][:, None]
    return (f >= e).long() + (f > c).long()  # 0 PRE, 1 BETWEEN (entry..collision), 2 POST


def random_crop_item(row, rng, min_len=256):
    """Context augmentation: a random window of a long clip that still contains ENTRY..COLLISION."""
    frames, _ = C.dense(row["sample_id"])
    n = len(frames)
    ei = int(np.searchsorted(frames, int(row["entry_frame"]))); ci = int(np.searchsorted(frames, int(row["collision_frame"])))
    lo = max(min_len, ci - ei + 1)
    if n <= lo: return C.make_item(row, "adaptive")
    length = int(rng.integers(lo, n + 1))
    start = int(rng.integers(max(0, ci - length + 1), min(ei, n - length) + 1))
    return C.make_item(row, "adaptive", start, start + length)


def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.inference_mode()
def evaluate(model, items, device, keep_logits=False):
    model.eval()
    return C.run(model, items, device, batch_size=1, keep_logits=keep_logits)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pool", choices=["avg", "max", "sgp"], default="avg")
    p.add_argument("--coarse", choices=["none", "ssm", "local_attn"], default="none")
    p.add_argument("--phase-weight", type=float, default=0.0)
    p.add_argument("--ema", type=float, default=0.0)
    p.add_argument("--crop-aug", type=float, default=0.0, help="probability of random context crop for clips > 256 frames")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--train-split", default="train")
    p.add_argument("--val-split", default="val")
    p.add_argument("--train-ids", default=None, help="optional json list restricting training clips (for OOF folds)")
    p.add_argument("--select-ids", default=None, help="optional json list of clips used for early stopping")
    p.add_argument("--output", default=None)
    a = p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    torch.set_num_threads(4)
    device = torch.device("cuda")
    out = Path(a.output) if a.output else RESULTS / a.run_id / f"seed{a.seed}"
    if (out / "checkpoint.pt").exists(): raise SystemExit(f"{out} exists; refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)

    train_rows = C.rows(a.train_split); val_rows = C.rows(a.val_split)
    if a.train_ids: keep = set(json.loads(Path(a.train_ids).read_text())); train_rows = [r for r in train_rows if r["sample_id"] in keep]
    select_rows = val_rows
    if a.select_ids: keep = set(json.loads(Path(a.select_ids).read_text())); select_rows = [r for r in C.rows("all") if r["sample_id"] in keep]
    train_fixed = [C.make_item(r, "adaptive") for r in train_rows]
    val_items = [C.make_item(r, "adaptive") for r in val_rows]
    select_items = val_items if not a.select_ids else [C.make_item(r, "adaptive") for r in select_rows]

    cfg = dict(vars(a), recipe="P2", lr=1e-3, weight_decay=.05, dropout=.35, hidden=128, token_dim=16, levels=4,
               event_target="soft_normalized", target_width=.015, target_width_mode="normalized_fixed",
               hard_negative_weight=0.0, token_dropout=0.0, channel_jitter=0.0, strict_fps_blind=True,
               n_train=len(train_rows), n_val=len(val_rows))
    model = LCPyramid(pool=a.pool, phase=3 if a.phase_weight else 0, coarse=a.coarse).to(device)
    params = sum(x.numel() for x in model.parameters())
    ema = copy.deepcopy(model).eval() if a.ema else None
    if ema is not None:
        for x in ema.parameters(): x.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.05)
    steps = (len(train_rows) + a.batch_size - 1) // a.batch_size
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=a.epochs * steps, pct_start=.1)
    (out / "config.json").write_text(json.dumps({**cfg, "parameter_count": params}, indent=2) + "\n")
    rng = np.random.default_rng(a.seed)
    gen = torch.Generator().manual_seed(a.seed)
    history, best, best_epoch, stale, best_state = [], -1.0, 0, 0, None
    torch.cuda.reset_peak_memory_stats(); wall = time.perf_counter()
    for epoch in range(1, a.epochs + 1):
        model.train(); total = 0.0; ph_total = 0.0
        order = torch.randperm(len(train_rows), generator=gen).tolist()
        for s in range(0, len(order), a.batch_size):
            idx = order[s:s + a.batch_size]
            items = [random_crop_item(train_rows[i], rng) if a.crop_aug and int(train_rows[i]["num_frames"]) > 256
                     and rng.random() < a.crop_aug else train_fixed[i] for i in idx]
            batch = to_device(collate(items), device)
            o = model(batch["x"], batch["time_valid"])
            loss, _ = experiment_loss(o, batch, cfg)
            if a.phase_weight:
                ph = F.cross_entropy(o["phase_logits"].float().transpose(1, 2), phase_targets(batch), reduction="none")
                ph = (ph * batch["time_valid"]).sum() / batch["time_valid"].sum()
                loss = loss + a.phase_weight * ph; ph_total += float(ph)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            if ema is not None:
                with torch.no_grad():
                    for t_, s_ in zip(ema.parameters(), model.parameters()): t_.lerp_(s_, 1 - a.ema)
            total += float(loss)
        m_eval = ema if ema is not None else model
        tr = normalized_metrics(evaluate(m_eval, train_fixed, device))
        sel_pred = evaluate(m_eval, select_items, device)
        va = normalized_metrics(sel_pred)
        score = va["fpsblind_selection_score"]
        rec = {"epoch": epoch, "train_loss": total / steps, "phase_loss": ph_total / steps,
               "train_fpsblind_score": tr["fpsblind_selection_score"], "val_fpsblind_score": score,
               "val_official": C.metrics(sel_pred)["score"]}
        history.append(rec); print(json.dumps(rec), flush=True)
        if score > best + 1e-8:
            best, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in m_eval.state_dict().items()}
        else:
            stale += 1
        if stale >= a.patience: break
    model.load_state_dict(best_state)
    t0 = time.perf_counter(); preds = evaluate(model, val_items, device, keep_logits=True); head_s = time.perf_counter() - t0
    res = C.breakdown(preds)
    rec_k = summarize_recall(recall_at_k(preds))
    train_score = normalized_metrics(evaluate(model, train_fixed, device))["fpsblind_selection_score"]
    summary = {"breakdown": res, "recall_at_k": rec_k, "best_epoch": best_epoch, "last_epoch": len(history),
               "train_fpsblind_at_best": train_score, "val_fpsblind_at_best": best,
               "runtime": {"wall_seconds": time.perf_counter() - wall, "head_ms_per_video": 1000 * head_s / len(preds),
                           "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20, "parameter_count": params}}
    torch.save({"model": best_state, "config": cfg}, out / "checkpoint.pt")
    C.dump(out / "history.json", history); C.dump(out / "metrics.json", summary)
    C.dump(out / "predictions.json", preds)
    print("RESULT", a.run_id, a.seed, C.short_table(res), f"| best_ep {best_epoch} params {params}", flush=True)


if __name__ == "__main__": main()
