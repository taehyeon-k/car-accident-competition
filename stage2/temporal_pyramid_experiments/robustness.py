"""Measure prediction and score sensitivity to FPS-blind sampling density."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from stage2.spotting_experiments.data import collate, read_rows
from stage2.spotting_experiments.models import build_model
from stage2.spotting_experiments.train import official_metrics, predict
from .data import PyramidFeatures
from .models import TemporalPyramid

B0 = Path("stage2/spotting_experiments/results/E3-ASFormer-349/fixed/seed0")
ROOT = Path("stage2/temporal_pyramid_experiments/results")
OLD = "stage2/artifacts/spotting_features/fpsblind_192_7x10_349"
NEW = "stage2/temporal_pyramid_experiments/cache_383"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", choices=["B0", "P1", "P2", "P3", "P4"])
    args = p.parse_args()
    if torch.cuda.is_available(): torch.cuda.set_per_process_memory_fraction(.05)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    rows = read_rows("/workspace/data/stage2/manifests/val.jsonl")
    all_rows = read_rows("/workspace/data/stage2/manifests/all.jsonl")
    for run in args.runs:
        path = B0 if run == "B0" else ROOT / run / "seed0"
        state = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=False)
        model = build_model(state["config"]) if run == "B0" else TemporalPyramid()
        model.load_state_dict(state["model"]); model.to(device).eval()
        counts = [96, 128, 192] if run in {"B0", "P1"} else [96, 128, 192, 256, 320]
        by_count = {}; head_ms = {}
        for count in counts:
            ds = PyramidFeatures(rows, OLD if run in {"B0", "P1"} else NEW,
                                 "P1" if run == "B0" else run, override_count=count)
            predictions, elapsed = predict(model, DataLoader(ds, batch_size=4, collate_fn=collate), device)
            by_count[str(count)] = {"predictions": predictions, "score": official_metrics(predictions, all_rows)}
            head_ms[str(count)] = 1000 * elapsed / len(predictions)
        trajectories = {}
        for event in ("entry", "collision"):
            values = np.asarray([[x[f"{event}_frame"] for x in by_count[str(c)]["predictions"]] for c in counts])
            spans = np.asarray([max(r["num_frames"] - 1, 1) for r in rows])
            trajectories[event] = {"mean_normalized_std": float(np.mean(values.std(0) / spans)),
                                   "median_original_frame_range": float(np.median(values.max(0) - values.min(0))),
                                   "mean_original_frame_range": float(np.mean(values.max(0) - values.min(0)))}
        scores = [by_count[str(c)]["score"]["competition_score"] for c in counts]
        result = {"counts": counts, "per_count": {str(c): by_count[str(c)]["score"] for c in counts},
                  "score_range": max(scores) - min(scores), "trajectories": trajectories,
                  "head_ms_per_video": head_ms}
        (path / "robustness.json").write_text(json.dumps(result, indent=2) + "\n")
        print(run, json.dumps(result))


if __name__ == "__main__": main()
