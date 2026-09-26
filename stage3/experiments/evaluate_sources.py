"""Evaluate one Stage 3 EMA checkpoint on named, unchanged full-clip manifests."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from sklearn.metrics import confusion_matrix

from stage3.trainer.decoder import ACCEL_LABELS, STEER_LABELS
from stage3.trainer.trainer import Trainer
from stage3.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", action="append", required=True,
                        help="name=/absolute/path/to/manifest.jsonl")
    parser.add_argument("--accel-threshold", type=float, default=.5)
    parser.add_argument("--gpu-memory-fraction", type=float, default=.55)
    args = parser.parse_args()
    if not 0 < args.gpu_memory_fraction < .6:
        parser.error("GPU memory fraction must be positive and below 0.6")
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, 0)
    cfg = load_config(args.config)
    cfg["decoder"]["acceleration"]["decelerating_below"] = -args.accel_threshold
    cfg["decoder"]["acceleration"]["accelerating_above"] = args.accel_threshold
    cfg["data"]["num_workers"] = 4  # I/O only; scores are full-clip, deterministic.
    output = {"checkpoint": str(args.checkpoint), "acceleration_threshold": args.accel_threshold,
              "steering_threshold": cfg["decoder"]["steering"]["threshold_deg"], "sources": {}}
    for spec in args.manifest:
        name, sep, path = spec.partition("=")
        if not sep or not name or not Path(path).is_file():
            parser.error(f"Invalid manifest specification: {spec}")
        current = copy.deepcopy(cfg)
        current["data"]["val_manifest"] = path
        trainer = Trainer(Accelerator(mixed_precision=current["optimization"]["mixed_precision"]), current)
        trainer.build()
        trainer.resume(str(args.checkpoint))
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        metrics = trainer.validate()
        pa, ta, ps, ts, am, sm = map(np.asarray, trainer.last_validation_predictions)
        sm = sm & (ta != "STOPPED")
        output["sources"][name] = {
            "manifest": path, "clips": len(trainer.val_set), "frames": len(ta),
            "seconds": time.time() - start,
            "peak_allocated_vram_mb": torch.cuda.max_memory_allocated() / 2**20,
            "metrics": metrics,
            "confusion_acceleration": confusion_matrix(ta[am], pa[am], labels=ACCEL_LABELS).tolist(),
            "confusion_steering": confusion_matrix(ts[sm], ps[sm], labels=STEER_LABELS).tolist(),
        }
        print(name, json.dumps({key: metrics[key] for key in
                                ("acceleration_macro_f1", "steering_macro_f1", "competition_score")}), flush=True)
        del trainer
        torch.cuda.empty_cache()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
