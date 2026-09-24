"""Evaluate a train-fitted, FPS-blind ENTRY→COLLISION sampled-index gap prior."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from stage2.spotting_experiments.data import collate, read_rows
from stage2.spotting_experiments.train import official_metrics, normalized_metrics, move
from .data import PyramidFeatures, adaptive_count
from .eval_ensemble import ROOT, ProbabilityEnsemble, load_member


def gap_parameters(rows):
    gaps = [(r["collision_frame"] - r["entry_frame"]) / max(r["num_frames"] - 1, 1) *
            (adaptive_count(r["num_frames"]) - 1) for r in rows]
    values = np.log(np.maximum(gaps, .5))
    return float(values.mean()), float(values.std())


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--id", required=True)
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--mode", choices=["gap", "collision_first", "entry_first"], default="gap")
    p.add_argument("members", nargs="+")
    args = p.parse_args()
    if torch.cuda.is_available(): torch.cuda.set_per_process_memory_fraction(.05)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_rows = read_rows("/workspace/data/stage2/manifests/all.jsonl")
    train_rows = read_rows("/workspace/data/stage2/manifests/train.jsonl")
    val_rows = read_rows("/workspace/data/stage2/manifests/val.jsonl")
    mu, sd = gap_parameters(train_rows)
    model = ProbabilityEnsemble([load_member(name, device) for name in args.members]).to(device).eval()
    loader = DataLoader(PyramidFeatures(val_rows, "stage2/temporal_pyramid_experiments/cache_383", "P2"),
                        batch_size=4, collate_fn=collate)
    predictions = []
    for original in loader:
        batch = move(original, device)
        outputs = model(batch["x"], batch["time_valid"])
        t = batch["x"].shape[1]
        indices = torch.arange(t, device=device)
        gap = indices[None, :] - indices[:, None]
        prior = -.5 * ((gap.clamp_min(1).float().log() - mu) / sd).square()
        if args.mode == "gap":
            pair = (outputs["entry_logits"].float()[:, :, None] +
                    outputs["collision_logits"].float()[:, None, :] +
                    args.strength * prior[None])
            valid_pair = ((gap > 0)[None] & batch["time_valid"][:, :, None] &
                          batch["time_valid"][:, None, :])
            pair = pair.masked_fill(~valid_pair, -1e9)
            flat = pair.flatten(1).argmax(-1)
            ei, ci = flat // t, flat % t
        elif args.mode == "collision_first":
            ci = outputs["collision_logits"].argmax(-1)
            admissible = (indices[None] <= ci[:, None]) & batch["time_valid"]
            ei = outputs["entry_logits"].masked_fill(~admissible, -1e9).argmax(-1)
        else:
            ei = outputs["entry_logits"].argmax(-1)
            admissible = (indices[None] >= ei[:, None]) & batch["time_valid"]
            ci = outputs["collision_logits"].masked_fill(~admissible, -1e9).argmax(-1)
        entry = batch["frame_numbers"].gather(1, ei[:, None]).squeeze(1)
        collision = batch["frame_numbers"].gather(1, ci[:, None]).squeeze(1)
        side = outputs["side_logits"].argmax(-1)
        evasion = (outputs["evasion_logits"] >= 0).long()
        for i, sample_id in enumerate(batch["sample_id"]):
            predictions.append({"sample_id": sample_id, "source_id": batch["source_id"][i],
                "entry_frame": int(entry[i]), "collision_frame": int(collision[i]),
                "entry_side": int(side[i]), "evasion_space": int(evasion[i]),
                "entry_gt": int(batch["entry_frame"][i]), "collision_gt": int(batch["collision_frame"][i]),
                "entry_side_gt": int(batch["entry_side"][i]), "evasion_gt": int(batch["evasion"][i]),
                "num_available_frames": int(batch["num_available_frames"][i])})
    metrics = {"official_offline": official_metrics(predictions, all_rows),
               "fpsblind": normalized_metrics(predictions),
               "members": args.members, "train_gap_log_mu": mu, "train_gap_log_sd": sd,
               "strength": args.strength, "mode": args.mode,
               "runtime": {"parameter_count": sum(p.numel() for p in model.parameters())}}
    out = ROOT / args.id / "seed0"; out.mkdir(parents=True, exist_ok=True)
    (out / "predictions.json").write_text(json.dumps(predictions, indent=2) + "\n")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__": main()
