"""Evaluate a predeclared ensemble of FPS-blind models on identical adaptive views."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from stage2.spotting_experiments.data import collate, read_rows
from stage2.spotting_experiments.models import build_model
from stage2.spotting_experiments.train import official_metrics, normalized_metrics, predict
from .data import PyramidFeatures
from .models import TemporalPyramid

ROOT = Path("stage2/temporal_pyramid_experiments/results")


class ProbabilityEnsemble(nn.Module):
    def __init__(self, models):
        super().__init__(); self.models = nn.ModuleList(models)

    def forward(self, x, valid):
        parts = [model(x, valid) for model in self.models]
        out = {}
        for event in ("entry", "collision"):
            probabilities = torch.stack([p[f"{event}_logits"].float().softmax(-1) for p in parts]).mean(0)
            out[f"{event}_logits"] = probabilities.clamp_min(1e-12).log()
        side = torch.stack([p["side_logits"].float().softmax(-1) for p in parts]).mean(0)
        evasion = torch.stack([p["evasion_logits"].float().sigmoid() for p in parts]).mean(0)
        out["side_logits"] = side.clamp_min(1e-12).log()
        out["evasion_logits"] = torch.logit(evasion.clamp(1e-6, 1 - 1e-6))
        return out


def load_member(name, device):
    state = torch.load(ROOT / name / "checkpoint.pt", map_location="cpu", weights_only=False)
    cfg = state["config"]
    if cfg.get("architecture") == "asformer": model = build_model(cfg)
    else: model = TemporalPyramid(global_coarse=cfg.get("global_coarse", False),
                                  category_head=cfg.get("category_head", "global"))
    model.load_state_dict(state["model"])
    return model.to(device).eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--id", required=True)
    p.add_argument("members", nargs="+")
    args = p.parse_args()
    if torch.cuda.is_available(): torch.cuda.set_per_process_memory_fraction(.05)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val = read_rows("/workspace/data/stage2/manifests/val.jsonl")
    all_rows = read_rows("/workspace/data/stage2/manifests/all.jsonl")
    ds = PyramidFeatures(val, "stage2/temporal_pyramid_experiments/cache_383", "P2")
    ensemble = ProbabilityEnsemble([load_member(x, device) for x in args.members]).to(device).eval()
    predictions, seconds = predict(ensemble, DataLoader(ds, batch_size=4, collate_fn=collate), device)
    metrics = {"official_offline": official_metrics(predictions, all_rows),
               "fpsblind": normalized_metrics(predictions),
               "members": args.members,
               "runtime": {"temporal_head_ms_per_video": 1000 * seconds / len(predictions),
                           "parameter_count": sum(p.numel() for p in ensemble.parameters())}}
    out = ROOT / args.id / "seed0"; out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (out / "predictions.json").write_text(json.dumps(predictions, indent=2) + "\n")
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__": main()
