"""Train one reproducible FPS-blind Stage-2 spotting run."""
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

from .data import SpottingFeatures, collate, read_rows
from .models import build_model
from .objective import decode, spotting_loss


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def source_name(value): return str(value).split(":", 1)[0]


def stratified_folds(rows, k=5, seed=42):
    strata = defaultdict(list)
    for row in rows: strata[(source_name(row.get("source_id", "")), row["entry_side"], row["evasion_space"])].append(row["sample_id"])
    rng = np.random.default_rng(seed); result = {}; offset = 0
    for key in sorted(strata):
        ids = sorted(strata[key]); rng.shuffle(ids)
        for i, sample_id in enumerate(ids): result[sample_id] = (i + offset) % k
        offset += len(ids)
    return result


def macro_f1(target, prediction):
    scores = []
    for cls in (0, 1):
        tp = sum(t == cls and p == cls for t, p in zip(target, prediction)); fp = sum(t != cls and p == cls for t, p in zip(target, prediction)); fn = sum(t == cls and p != cls for t, p in zip(target, prediction))
        scores.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(scores))


def normalized_metrics(predictions):
    ee = np.asarray([abs(x["entry_frame"] - x["entry_gt"]) / max(x["num_available_frames"] - 1, 1) for x in predictions])
    ce = np.asarray([abs(x["collision_frame"] - x["collision_gt"]) / max(x["num_available_frames"] - 1, 1) for x in predictions])
    result = {"entry_mae_frames": float(np.mean([abs(x["entry_frame"] - x["entry_gt"]) for x in predictions])),
              "collision_mae_frames": float(np.mean([abs(x["collision_frame"] - x["collision_gt"]) for x in predictions])),
              "entry_normalized_mae": float(ee.mean()), "collision_normalized_mae": float(ce.mean()),
              "entry_normalized_median": float(np.median(ee)), "collision_normalized_median": float(np.median(ce))}
    for threshold in (.005, .01, .02, .05):
        key = str(threshold * 100).rstrip("0").rstrip(".").replace(".", "p")
        result[f"entry_p_within_{key}pct"] = float((ee <= threshold).mean()); result[f"collision_p_within_{key}pct"] = float((ce <= threshold).mean())
    side = macro_f1([x["entry_side_gt"] for x in predictions], [x["entry_side"] for x in predictions])
    evasion = macro_f1([x["evasion_gt"] for x in predictions], [x["evasion_space"] for x in predictions])
    result.update(f1_entry_side_macro=side, f1_evasion_space_macro=evasion)
    result["fpsblind_selection_score"] = .35 * result["entry_p_within_1pct"] + .35 * result["collision_p_within_1pct"] + .15 * side + .15 * evasion
    return result


def official_metrics(predictions, rows):
    """Offline evaluator: FPS is first accessed after prediction is complete."""
    metadata = {x["sample_id"]: x for x in rows}
    entry = []; collision = []
    for pred in predictions:
        row = metadata[pred["sample_id"]]; fps = float(row["native_fps"])
        entry.append(abs(pred["entry_frame"] - pred["entry_gt"]) / fps <= .300001)
        collision.append(abs(pred["collision_frame"] - pred["collision_gt"]) / fps <= .300001)
    side = macro_f1([x["entry_side_gt"] for x in predictions], [x["entry_side"] for x in predictions])
    evasion = macro_f1([x["evasion_gt"] for x in predictions], [x["evasion_space"] for x in predictions])
    result = {"acc_entry_0.3s": float(np.mean(entry)), "acc_collision_0.3s": float(np.mean(collision)),
              "f1_entry_side_macro": side, "f1_evasion_space_macro": evasion}
    result["competition_score"] = .35 * result["acc_entry_0.3s"] + .35 * result["acc_collision_0.3s"] + .15 * side + .15 * evasion
    return result


def move(batch, device): return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.inference_mode()
def predict(model, loader, device):
    model.eval(); predictions = []; head_seconds = 0.0
    for batch in loader:
        batch = move(batch, device)
        if device.type == "cuda": torch.cuda.synchronize()
        start = time.perf_counter(); outputs = model(batch["x"], batch["time_valid"])
        if device.type == "cuda": torch.cuda.synchronize()
        head_seconds += time.perf_counter() - start
        entry, collision = decode(outputs, batch); side = outputs["side_logits"].argmax(-1); evasion = (outputs["evasion_logits"] >= 0).long()
        for i, sample_id in enumerate(batch["sample_id"]):
            predictions.append({"sample_id": sample_id, "source_id": batch["source_id"][i],
                "entry_frame": int(entry[i]), "collision_frame": int(collision[i]),
                "entry_side": int(side[i]), "evasion_space": int(evasion[i]),
                "entry_gt": int(batch["entry_frame"][i]), "collision_gt": int(batch["collision_frame"][i]),
                "entry_side_gt": int(batch["entry_side"][i]), "evasion_gt": int(batch["evasion"][i]),
                "num_available_frames": int(batch["num_available_frames"][i])})
    return predictions, head_seconds


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int); parser.add_argument("--fold", type=int, default=-1)
    args = parser.parse_args(); cfg = json.loads(Path(args.config).read_text()); seed = cfg["seed"] if args.seed is None else args.seed
    if not cfg.get("strict_fps_blind", False): raise ValueError("This trainer is intentionally strict; use downstream_probe.py for E0-current")
    seed_all(seed); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_rows = read_rows(cfg["all_manifest"]); fixed_train = read_rows(cfg["train_manifest"]); fixed_val = read_rows(cfg["val_manifest"])
    if args.fold >= 0:
        fold_of = stratified_folds(all_rows, cfg.get("folds", 5), cfg.get("fold_seed", 42)); train_rows = [x for x in all_rows if fold_of[x["sample_id"]] != args.fold]; val_rows = [x for x in all_rows if fold_of[x["sample_id"]] == args.fold]
    else: train_rows, val_rows = fixed_train, fixed_val
    ds_args = dict(cache_dir=cfg["cache_dir"], strict_fps_blind=True)
    train_data = SpottingFeatures(train_rows, sample_counts=cfg["train_sample_counts"], train=True, jitter=cfg.get("jitter", 0), drop_probability=cfg.get("drop_probability", 0), seed=seed, **ds_args)
    val_data = SpottingFeatures(val_rows, sample_counts=[cfg["eval_sample_count"]], train=False, seed=seed, **ds_args)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_data, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate, num_workers=cfg.get("workers", 0), generator=generator)
    val_loader = DataLoader(val_data, batch_size=cfg.get("eval_batch_size", 16), collate_fn=collate, num_workers=cfg.get("workers", 0))
    model = build_model(cfg).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg["lr"], total_steps=cfg["epochs"] * len(train_loader), pct_start=.1)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True); history = []; best_score = -1.; best_state = None
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats()
    wall_start = time.perf_counter()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train(); sums = defaultdict(float)
        for batch in train_loader:
            batch = move(batch, device); outputs = model(batch["x"], batch["time_valid"]); loss, parts = spotting_loss(outputs, batch, cfg)
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0)); optimizer.step(); scheduler.step()
            sums["loss"] += float(loss); [sums.__setitem__(k, sums[k] + float(v)) for k, v in parts.items()]
        predictions, _ = predict(model, val_loader, device); metrics = normalized_metrics(predictions)
        record = {"epoch": epoch, **{k: v / len(train_loader) for k, v in sums.items()}, **metrics}; history.append(record)
        if metrics["fpsblind_selection_score"] > best_score:
            best_score = metrics["fpsblind_selection_score"]; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(json.dumps(record), flush=True)
    model.load_state_dict(best_state); predictions, head_seconds = predict(model, val_loader, device)
    fpsblind = normalized_metrics(predictions); official = official_metrics(predictions, all_rows)
    runtime = {"wall_seconds": time.perf_counter() - wall_start, "temporal_head_seconds_validation": head_seconds,
               "temporal_head_ms_per_video": head_seconds / len(predictions) * 1000,
               "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0,
               "parameter_count": sum(p.numel() for p in model.parameters())}
    try: commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception: commit = "unknown"
    state = {"model": best_state, "config": cfg, "seed": seed, "fold": args.fold, "fpsblind_metrics": fpsblind,
             "official_offline_metrics": official, "sampling_version": "strict-fps-blind-v1"}
    torch.save(state, out / "checkpoint.pt")
    metadata = {"config": cfg, "config_path": str(Path(args.config).resolve()), "git_commit": commit, "seed": seed, "fold": args.fold,
                "train_ids": [x["sample_id"] for x in train_rows], "val_ids": [x["sample_id"] for x in val_rows]}
    (out / "config.json").write_text(json.dumps(metadata, indent=2) + "\n"); (out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (out / "predictions.json").write_text(json.dumps(predictions, indent=2) + "\n")
    metrics = {"fpsblind": fpsblind, "official_offline": official, "runtime": runtime}; (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__": main()
