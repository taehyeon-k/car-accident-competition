"""Refit a P2 pyramid seed on a whole manifest for a fixed number of epochs (no validation).

Identical to train.py for P2 (data, model, loss, AdamW, OneCycle schedule sized for the
original 30-epoch budget, batch size, seed), except that training stops after the epoch
count that validation selected for this seed, and no validation split is held out.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from stage2.spotting_experiments.data import collate, read_rows
from stage2.spotting_experiments.train import move, normalized_metrics, predict
from .data import PyramidFeatures
from .models import TemporalPyramid
from .train import augment_features, experiment_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="/workspace/data/stage2/manifests/all.jsonl")
    p.add_argument("--cache", default="stage2/temporal_pyramid_experiments/cache_383")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--stop-epoch", type=int, required=True, help="epoch selected by validation for this seed")
    p.add_argument("--schedule-epochs", type=int, default=30, help="OneCycle length, as in the validated run")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        free, _ = torch.cuda.mem_get_info()
        if free < 3 * 2**30: raise RuntimeError("Less than 3 GiB free; preserving active GPU jobs")
        torch.cuda.set_per_process_memory_fraction(.05)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    rows = read_rows(args.manifest)
    ds_kwargs = dict(cache_dir=args.cache, experiment="P2", seed=args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(PyramidFeatures(rows, train=True, **ds_kwargs), batch_size=4, shuffle=True,
                              num_workers=0, collate_fn=collate, generator=generator)
    cfg = {"experiment": "P2", "seed": args.seed, "epochs": args.schedule_epochs, "stop_epoch": args.stop_epoch,
           "refit_manifest": args.manifest, "n_videos": len(rows), "batch_size": 4, "optimizer": "AdamW",
           "lr": .001, "weight_decay": .05, "dropout": .35, "hidden": 128, "token_dim": 16, "levels": 4,
           "cache": args.cache, "global_coarse": False, "long_oversample": 1.0, "category_head": "global",
           "architecture": "pyramid", "event_target": "soft_normalized", "target_width": .015,
           "target_width_mode": "normalized_fixed", "hard_negative_weight": 0.0, "ema_decay": 0.0,
           "token_dropout": 0.0, "channel_jitter": 0.0, "strict_fps_blind": True, "consistency_weight": 0}
    model = TemporalPyramid().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.05)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=.001,
                    total_steps=args.schedule_epochs * len(train_loader), pct_start=.1)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    history, wall = [], time.perf_counter()
    for epoch in range(1, args.stop_epoch + 1):
        model.train(); total = 0.0
        for data in train_loader:
            a = augment_features(move(data, device), cfg)
            loss, _ = experiment_loss(model(a["x"], a["time_valid"]), a, cfg)
            optimizer.zero_grad(set_to_none=True)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            total += float(loss.detach())
        history.append({"epoch": epoch, "train_loss": total / len(train_loader), "lr": scheduler.get_last_lr()[0]})
        print(json.dumps(history[-1]), flush=True)
    model.eval()
    fit_pred, _ = predict(model, DataLoader(PyramidFeatures(rows, **ds_kwargs), batch_size=4, collate_fn=collate), device)
    metrics = {"in_sample_fpsblind": normalized_metrics(fit_pred), "epochs_trained": args.stop_epoch,
               "wall_seconds": time.perf_counter() - wall,
               "note": "in-sample fit only; no held-out validation exists for a full refit"}
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save({"model": state, "config": cfg, "metrics": metrics}, out / "checkpoint.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({"done": str(out), "in_sample_score": metrics["in_sample_fpsblind"]["fpsblind_selection_score"]}), flush=True)


if __name__ == "__main__": main()
