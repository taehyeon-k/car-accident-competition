"""CPU-only same-view evaluation of the saved 349-video baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from stage2.spotting_experiments.data import SpottingFeatures, collate, read_rows
from stage2.spotting_experiments.models import build_model
from stage2.spotting_experiments.train import official_metrics, normalized_metrics, predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=192)
    args = parser.parse_args()
    torch.set_num_threads(4)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = state["config"]
    rows = read_rows(cfg["val_manifest"])
    all_rows = read_rows(cfg["all_manifest"])
    dataset = SpottingFeatures(rows, cfg["cache_dir"], sample_counts=(args.count,),
                               strict_fps_blind=True)
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate, num_workers=0)
    model = build_model(cfg).eval()
    model.load_state_dict(state["model"])
    predictions, _ = predict(model, loader, torch.device("cpu"))
    result = {"count": args.count, "fpsblind": normalized_metrics(predictions),
              "official_diagnostic": official_metrics(predictions, all_rows)}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "predictions.json").write_text(json.dumps(predictions, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
