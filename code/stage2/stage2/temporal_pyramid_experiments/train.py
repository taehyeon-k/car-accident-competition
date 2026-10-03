"""Train P1–P4 on the same 349-video split with early stopping."""
from __future__ import annotations

import argparse
import copy
import json
import random
import faulthandler
import signal
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from stage2.spotting_experiments.data import collate, read_rows
from stage2.spotting_experiments.objective import spotting_loss
from stage2.spotting_experiments.models import build_model
from stage2.spotting_experiments.train import move, normalized_metrics, official_metrics, predict
from .data import PyramidFeatures
from .models import TemporalPyramid


class Paired(Dataset):
    def __init__(self, a, b): self.a, self.b = a, b
    def __len__(self): return len(self.a)
    def __getitem__(self, i): return self.a[i], self.b[i]


def paired_collate(items):
    return collate([x[0] for x in items]), collate([x[1] for x in items])


class ExtraLongSampler(Sampler):
    """See every training video once, then add extra long-video draws."""
    def __init__(self, rows, factor, generator):
        self.size = len(rows)
        self.long_indices = torch.tensor([i for i, row in enumerate(rows) if int(row["num_frames"]) > 500])
        self.extra = round((factor - 1) * len(self.long_indices))
        self.generator = generator

    def __len__(self): return self.size + self.extra

    def __iter__(self):
        base = torch.randperm(self.size, generator=self.generator)
        extras = self.long_indices[torch.randint(len(self.long_indices), (self.extra,), generator=self.generator)]
        merged = torch.cat([base, extras])
        yield from merged[torch.randperm(len(merged), generator=self.generator)].tolist()


def interpolated_cdf(logits, positions, valid, grid):
    probabilities = logits.float().masked_fill(~valid, -1e4).softmax(-1)
    cdf = probabilities.cumsum(-1)
    positions = positions.float()
    upper = torch.searchsorted(positions.contiguous(), grid.expand(len(positions), -1).contiguous()).clamp(1, positions.shape[1] - 1)
    lower = upper - 1
    pl = positions.gather(1, lower); pu = positions.gather(1, upper)
    cl = cdf.gather(1, lower); cu = cdf.gather(1, upper)
    fraction = ((grid - pl) / (pu - pl).clamp_min(1e-6)).clamp(0, 1)
    return cl + fraction * (cu - cl)


def consistency(a, ba, b, bb):
    grid = torch.linspace(0, 1, 129, device=ba["x"].device)[None]
    return sum((interpolated_cdf(a[f"{event}_logits"], ba["normalized_positions"], ba["time_valid"], grid) -
                interpolated_cdf(b[f"{event}_logits"], bb["normalized_positions"], bb["time_valid"], grid)).abs().mean()
               for event in ("entry", "collision")) / 2


def experiment_loss(outputs, batch, cfg):
    loss_cfg = cfg
    if cfg["target_width_mode"] == "adaptive_index2":
        # Keep the original normalized width for sparse/short clips, but cap
        # dense views at about two sampled positions. No FPS is involved.
        counts = batch["time_valid"].sum(-1, keepdim=True).float()
        width = torch.minimum(torch.full_like(counts, .015), 2.0 / (counts - 1).clamp_min(1))
        loss_cfg = {**cfg, "target_width": width}
    loss, parts = spotting_loss(outputs, batch, loss_cfg)
    if cfg["hard_negative_weight"]:
        positions = batch["normalized_positions"]
        event_losses = []
        for event in ("entry", "collision"):
            logits = outputs[f"{event}_logits"].float()
            index = batch[f"{event}_index"][:, None]
            positive = logits.gather(1, index).squeeze(1)
            target = positions.gather(1, index)
            outside = batch["time_valid"] & ((positions - target).abs() > .03)
            hard = logits.masked_fill(~outside, -1e4).topk(min(3, logits.shape[1]), -1).values.mean(-1)
            event_losses.append(F.relu(.5 + hard - positive).mean())
        hard_loss = sum(event_losses) / len(event_losses)
        loss = loss + cfg["hard_negative_weight"] * hard_loss
        parts["hard_negative"] = hard_loss.detach()
    return loss, parts


def augment_features(batch, cfg):
    if not (cfg["token_dropout"] or cfg["channel_jitter"]): return batch
    x = batch["x"].float()
    if cfg["channel_jitter"]:
        gain = 1 + cfg["channel_jitter"] * torch.randn(
            x.shape[0], 1, 1, x.shape[-1], device=x.device)
        x = x * gain
    if cfg["token_dropout"]:
        keep = (torch.rand(*x.shape[:-1], 1, device=x.device) >= cfg["token_dropout"]).float()
        x = x * keep
    return {**batch, "x": x.half()}


def main():
    faulthandler.register(signal.SIGUSR1, all_threads=False)
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", choices=["P1", "P2", "P3", "P4"], required=True)
    p.add_argument("--old-cache", default="stage2/artifacts/spotting_features/fpsblind_192_7x10_349")
    p.add_argument("--new-cache", default="stage2/temporal_pyramid_experiments/cache_383")
    p.add_argument("--output", default="stage2/temporal_pyramid_experiments/results")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-id", default=None)
    p.add_argument("--global-coarse", action="store_true")
    p.add_argument("--long-oversample", type=float, default=1.0)
    p.add_argument("--category-head", choices=["global", "event"], default="global")
    p.add_argument("--architecture", choices=["pyramid", "asformer"], default="pyramid")
    p.add_argument("--target-width-mode", choices=["normalized_fixed", "adaptive_index2"], default="normalized_fixed")
    p.add_argument("--hard-negative-weight", type=float, default=0.0)
    p.add_argument("--ema-decay", type=float, default=0.0)
    p.add_argument("--token-dropout", type=float, default=0.0)
    p.add_argument("--channel-jitter", type=float, default=0.0)
    args = p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        free, _ = torch.cuda.mem_get_info()
        if free < 3 * 2**30: raise RuntimeError("Less than 3 GiB free; preserving active GPU jobs")
        torch.cuda.set_per_process_memory_fraction(.05)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    rows = read_rows("/workspace/data/stage2/manifests/all.jsonl")
    train_rows = read_rows("/workspace/data/stage2/manifests/train.jsonl")
    val_rows = read_rows("/workspace/data/stage2/manifests/val.jsonl")
    cache = args.old_cache if args.experiment == "P1" else args.new_cache
    ds_kwargs = dict(cache_dir=cache, experiment=args.experiment, seed=args.seed)
    train_ds = PyramidFeatures(train_rows, train=True, **ds_kwargs)
    if args.experiment == "P4":
        train_ds = Paired(train_ds, PyramidFeatures(train_rows, train=True, view=1, **ds_kwargs))
    generator = torch.Generator().manual_seed(args.seed)
    sampler = (ExtraLongSampler(train_rows, args.long_oversample, generator)
               if args.long_oversample != 1.0 else None)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=sampler is None,
                              sampler=sampler, num_workers=0,
                              collate_fn=paired_collate if args.experiment == "P4" else collate,
                              generator=generator)
    train_eval = DataLoader(PyramidFeatures(train_rows, **ds_kwargs), batch_size=args.batch_size, collate_fn=collate)
    val_eval = DataLoader(PyramidFeatures(val_rows, **ds_kwargs), batch_size=args.batch_size, collate_fn=collate)
    cfg = {"experiment": args.experiment, "seed": args.seed, "epochs": args.epochs,
           "patience": args.patience, "batch_size": args.batch_size, "optimizer": "AdamW",
           "lr": .001, "weight_decay": .05, "dropout": .35, "hidden": 128,
           "token_dim": 16, "levels": 4, "cache": cache,
           "global_coarse": args.global_coarse, "long_oversample": args.long_oversample,
           "category_head": args.category_head,
           "architecture": args.architecture,
           "event_target": "soft_normalized", "target_width": .015,
           "target_width_mode": args.target_width_mode,
           "hard_negative_weight": args.hard_negative_weight,
           "ema_decay": args.ema_decay,
           "token_dropout": args.token_dropout, "channel_jitter": args.channel_jitter,
           "strict_fps_blind": True, "consistency_weight": .05 if args.experiment == "P4" else 0}
    if args.architecture == "asformer":
        cfg.update(head="asformer", feature_dim=384, n_tokens=70, token_dim=32,
                   hidden=192, dropout=.3, attention_layers=2, attention_heads=4,
                   attention_window=17, category_head="global", offsets=False)
        model = build_model(cfg).to(device)
    else:
        model = TemporalPyramid(global_coarse=args.global_coarse,
                                category_head=args.category_head).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ema_model = copy.deepcopy(model).eval() if args.ema_decay else None
    if ema_model is not None:
        for parameter in ema_model.parameters(): parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.05)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=.001,
                       total_steps=args.epochs * len(train_loader), pct_start=.1)
    out = Path(args.output) / (args.run_id or args.experiment) / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    history, best_score, best_epoch, stale = [], -1.0, 0, 0
    best_state = None
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats()
    wall = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0; consistency_total = 0.0
        epoch_start = time.perf_counter()
        for step, data in enumerate(train_loader):
            if epoch == 1 and step < 5:
                print(json.dumps({"experiment": args.experiment, "starting_batch": step + 1,
                                  "length": int((data[0] if args.experiment == "P4" else data)["x"].shape[1]),
                                  "elapsed_seconds": round(time.perf_counter() - epoch_start, 2)}), flush=True)
            if args.experiment == "P4":
                a, b = (move(x, device) for x in data)
                a, b = augment_features(a, cfg), augment_features(b, cfg)
                oa, ob = model(a["x"], a["time_valid"]), model(b["x"], b["time_valid"])
                la, _ = experiment_loss(oa, a, cfg); lb, _ = experiment_loss(ob, b, cfg)
                c = consistency(oa, a, ob, b)
                loss = (la + lb) / 2 + .05 * c
                consistency_total += float(c.detach())
            else:
                a = augment_features(move(data, device), cfg)
                oa = model(a["x"], a["time_valid"])
                loss, _ = experiment_loss(oa, a, cfg)
            optimizer.zero_grad(set_to_none=True)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            if ema_model is not None:
                with torch.no_grad():
                    for teacher, student in zip(ema_model.parameters(), model.parameters()):
                        teacher.lerp_(student, 1 - args.ema_decay)
            total += float(loss.detach())
            if epoch == 1 and (step + 1) % 5 == 0:
                print(json.dumps({"experiment": args.experiment, "epoch": epoch,
                                  "batch": step + 1, "total_batches": len(train_loader),
                                  "elapsed_seconds": round(time.perf_counter() - epoch_start, 2),
                                  "length": int(a["x"].shape[1])}), flush=True)
        eval_model = ema_model if ema_model is not None else model
        train_pred, _ = predict(eval_model, train_eval, device)
        val_pred, _ = predict(eval_model, val_eval, device)
        train_met = normalized_metrics(train_pred); val_met = normalized_metrics(val_pred)
        score = val_met["fpsblind_selection_score"]
        rec = {"epoch": epoch, "train_loss": total / len(train_loader),
               "train_fpsblind_score": train_met["fpsblind_selection_score"],
               "val_fpsblind_score": score,
               "val_entry_1pct": val_met["entry_p_within_1pct"],
               "val_collision_1pct": val_met["collision_p_within_1pct"],
               "val_side_f1": val_met["f1_entry_side_macro"],
               "val_evasion_f1": val_met["f1_evasion_space_macro"],
               "consistency": consistency_total / len(train_loader) if args.experiment == "P4" else 0}
        history.append(rec)
        if score > best_score + 1e-8:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in eval_model.state_dict().items()}
        else: stale += 1
        (out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps(rec), flush=True)
        if stale >= args.patience: break
    model.load_state_dict(best_state)
    val_pred, head_seconds = predict(model, val_eval, device)
    train_pred, _ = predict(model, train_eval, device)
    metrics = {"fpsblind": normalized_metrics(val_pred), "official_offline": official_metrics(val_pred, rows),
               "train_fpsblind": normalized_metrics(train_pred),
               "runtime": {"wall_seconds": time.perf_counter() - wall,
                           "temporal_head_ms_per_video": 1000 * head_seconds / len(val_pred),
                           "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0,
                           "parameter_count": params, "eval_encoded_frames": "adaptive rule" if args.experiment != "P1" else 128},
               "best_epoch": best_epoch, "last_epoch": len(history),
               "best_to_final_epoch_gap": len(history) - best_epoch}
    torch.save({"model": best_state, "config": cfg, "metrics": metrics}, out / "checkpoint.pt")
    (out / "predictions.json").write_text(json.dumps(val_pred, indent=2) + "\n")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__": main()
