"""Sampling-density robustness (prompt v2 section 18).

Re-evaluates one trained checkpoint under several temporal sampling schemes,
maps every prediction back to original-frame coordinates, and reports how much
the predicted event moves. The private data has unknown FPS, so a good model
should barely move.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import SpottingViews, collate, read_rows
from .metrics import official_metrics
from .models import build_model
from .train import predict

# Each scheme is a sampling configuration, not a time base.
SCHEMES = {
    "dense_192": dict(sample_counts=[192]),
    "every_2nd_96": dict(sample_counts=[96]),
    "standard_128": dict(sample_counts=[128]),
    "drop_20pct": dict(sample_counts=[128], drop_probability=0.20),
    "drop_40pct": dict(sample_counts=[128], drop_probability=0.40),
    "irregular_warp": dict(sample_counts=[128], warp=0.35, jitter=0.6),
}


def evaluate_scheme(model, rows, cache_dir, device, seed, **kwargs):
    counts = kwargs.pop("sample_counts")
    data = SpottingViews(rows, cache_dir=cache_dir, sample_counts=counts, train=False,
                         seed=seed, **kwargs)
    # Evaluation is deterministic, so force the augmentation through explicitly.
    data.train = True
    data.sample_counts = tuple(counts)
    loader = DataLoader(data, batch_size=16, collate_fn=collate)
    predictions, _ = predict(model, loader, device)
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = state["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    model.load_state_dict(state["model"])

    all_rows = read_rows(cfg["all_manifest"])
    val_rows = read_rows(cfg["val_manifest"])

    per_scheme, positions = {}, {}
    for name, kwargs in SCHEMES.items():
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        predictions = evaluate_scheme(model, val_rows, cfg["cache_dir"], device,
                                      args.seed, **dict(kwargs))
        per_scheme[name] = official_metrics(predictions, all_rows)
        for pred in predictions:
            span = max(pred["num_available_frames"] - 1, 1)
            slot = positions.setdefault(pred["sample_id"], {"entry": [], "collision": []})
            slot["entry"].append(pred["entry_frame"] / span)
            slot["collision"].append(pred["collision_frame"] / span)

    instability = {}
    for event in ("entry", "collision"):
        spreads = np.asarray([np.std(v[event]) for v in positions.values()])
        ranges = np.asarray([np.ptp(v[event]) for v in positions.values()])
        instability[event] = {
            "mean_normalized_std": float(spreads.mean()),
            "median_normalized_std": float(np.median(spreads)),
            "mean_normalized_range": float(ranges.mean()),
            "p90_normalized_range": float(np.percentile(ranges, 90)),
        }

    scores = [m["competition_score"] for m in per_scheme.values()]
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "config_id": cfg.get("id"),
        "per_scheme": per_scheme,
        "instability": instability,
        "score_spread": {"min": float(min(scores)), "max": float(max(scores)),
                         "range": float(max(scores) - min(scores))},
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
