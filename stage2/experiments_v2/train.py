"""Train one reproducible FPS-blind Stage-2 run from the v2 experiment matrix."""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import SpottingViews, collate, read_rows
from .metrics import by_source, fpsblind_metrics, official_metrics
from .models import build_model
from .objective import consistency_loss, decode, spotting_loss


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    return value


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    predictions, head_seconds = [], 0.0
    for batch in loader:
        batch = move(batch, device)
        view = batch["a"]
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = model(view["x"], view["time_valid"])
        if device.type == "cuda":
            torch.cuda.synchronize()
        head_seconds += time.perf_counter() - start
        entry, collision = decode(outputs, view)
        side = outputs["side_logits"].argmax(-1)
        evasion = (outputs["evasion_logits"] >= 0).long()
        for i, sample_id in enumerate(batch["sample_id"]):
            predictions.append({
                "sample_id": sample_id, "source_id": batch["source_id"][i],
                "entry_frame": int(entry[i]), "collision_frame": int(collision[i]),
                "entry_side": int(side[i]), "evasion_space": int(evasion[i]),
                "entry_gt": int(batch["entry_frame"][i]),
                "collision_gt": int(batch["collision_frame"][i]),
                "entry_side_gt": int(batch["entry_side"][i]),
                "evasion_gt": int(batch["evasion"][i]),
                "num_available_frames": int(batch["num_available_frames"][i]),
            })
    return predictions, head_seconds


def build_loaders(cfg, seed, train_rows, val_rows):
    common = dict(cache_dir=cfg["cache_dir"], seed=seed)
    train_data = SpottingViews(
        train_rows, sample_counts=cfg["train_sample_counts"], train=True,
        jitter=cfg.get("jitter", 0.0), drop_probability=cfg.get("drop_probability", 0.0),
        warp=cfg.get("warp", 0.0), duplicate_probability=cfg.get("duplicate_probability", 0.0),
        two_views=bool(cfg.get("consistency_weight", 0.0)), **common)
    val_data = SpottingViews(val_rows, sample_counts=[cfg["eval_sample_count"]],
                             train=False, **common)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_data, batch_size=cfg["batch_size"], shuffle=True,
                              collate_fn=collate, num_workers=cfg.get("workers", 0),
                              generator=generator, drop_last=False)
    val_loader = DataLoader(val_data, batch_size=cfg.get("eval_batch_size", 16),
                            collate_fn=collate, num_workers=cfg.get("workers", 0))
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    seed = cfg.get("seed", 42) if args.seed is None else args.seed
    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_rows = read_rows(cfg["all_manifest"])
    train_rows = read_rows(cfg["train_manifest"])
    val_rows = read_rows(cfg["val_manifest"])
    train_loader, val_loader = build_loaders(cfg, seed, train_rows, val_rows)

    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                  weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg["lr"], total_steps=cfg["epochs"] * len(train_loader), pct_start=0.1)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    consistency_weight = float(cfg.get("consistency_weight", 0.0))
    history, best_score, best_state = [], -1.0, None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    wall_start = time.perf_counter()

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        sums = defaultdict(float)
        for batch in train_loader:
            batch = move(batch, device)
            outputs = model(batch["a"]["x"], batch["a"]["time_valid"])
            loss, parts = spotting_loss(outputs, batch, batch["a"], cfg)
            if consistency_weight and "b" in batch:
                out_b = model(batch["b"]["x"], batch["b"]["time_valid"])
                loss_b, _ = spotting_loss(out_b, batch, batch["b"], cfg)
                agree = consistency_loss(outputs, batch["a"], out_b, batch["b"],
                                         int(cfg.get("consistency_bins", 64))).mean()
                # Both views are supervised; the agreement term is the extra signal.
                loss = 0.5 * (loss + loss_b) + consistency_weight * agree
                parts["loss_consistency"] = agree
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
            optimizer.step()
            scheduler.step()
            sums["loss"] += float(loss.detach())
            for k, v in parts.items():
                sums[k] += float(v)
        predictions, _ = predict(model, val_loader, device)
        metrics = fpsblind_metrics(predictions)
        record = {"epoch": epoch, **{k: v / len(train_loader) for k, v in sums.items()},
                  "fpsblind_selection_score": metrics["fpsblind_selection_score"],
                  "entry_p_within_1pct": metrics["entry_p_within_1pct"],
                  "collision_p_within_1pct": metrics["collision_p_within_1pct"]}
        history.append(record)
        if metrics["fpsblind_selection_score"] > best_score:
            best_score = metrics["fpsblind_selection_score"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(json.dumps(record), flush=True)

    model.load_state_dict(best_state)
    predictions, head_seconds = predict(model, val_loader, device)
    fpsblind = fpsblind_metrics(predictions)
    official = official_metrics(predictions, all_rows)
    runtime = {
        "wall_seconds": time.perf_counter() - wall_start,
        "head_seconds_validation": head_seconds,
        "head_ms_per_video": head_seconds / max(len(predictions), 1) * 1000,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else 0,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"

    torch.save({"model": best_state, "config": cfg, "seed": seed,
                "fpsblind_metrics": fpsblind, "official_offline_metrics": official,
                "sampling_version": "experiments-v2"}, out / "checkpoint.pt")
    (out / "config.json").write_text(json.dumps(
        {"config": cfg, "config_path": str(Path(args.config).resolve()), "git_commit": commit,
         "seed": seed, "train_ids": [x["sample_id"] for x in train_rows],
         "val_ids": [x["sample_id"] for x in val_rows]}, indent=2) + "\n")
    (out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (out / "predictions.json").write_text(json.dumps(predictions, indent=2) + "\n")
    metrics = {"fpsblind": fpsblind, "official_offline": official, "runtime": runtime,
               "by_source": by_source(predictions, all_rows)}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({"id": cfg.get("id"), **official, "params": runtime["parameter_count"]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
