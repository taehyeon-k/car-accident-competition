"""Measure Stage 2 inference in the current environment; no GPU is required."""

import argparse
import json
import time

import torch

from stage2.model.inference import Stage2Pipeline
from stage2.utils.checkpoint import load_inference_system
from stage2.utils.utils import read_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coarse-ckpt",
        required=True,
    )
    parser.add_argument(
        "--fine-ckpt",
        required=True,
    )
    parser.add_argument(
        "--manifest",
        required=True,
    )
    parser.add_argument(
        "--device",
        default="cpu",
    )
    arguments = parser.parse_args()
    coarse, config = load_inference_system(
        arguments.coarse_ckpt,
        "coarse",
    )
    fine, _ = load_inference_system(
        arguments.fine_ckpt,
        "fine",
    )
    pipeline = Stage2Pipeline(
        coarse,
        fine,
        config,
        arguments.device,
    )
    if torch.device(arguments.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    rows = read_manifest(arguments.manifest)
    for row in rows:
        pipeline.predict(row)
    if torch.device(arguments.device).type == "cuda":
        torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "stage": 2,
                "samples": len(rows),
                "elapsed_seconds": time.perf_counter() - start,
                "device": arguments.device,
                "peak_cuda_bytes": (
                    torch.cuda.max_memory_allocated()
                    if torch.device(arguments.device).type == "cuda"
                    else None
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
