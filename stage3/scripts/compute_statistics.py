from __future__ import annotations

import argparse

import torch

from stage3.utils.checkpoint import atomic_save
from stage3.utils.checkpoint import load_artifact
from stage3.utils.config import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute training-only robust physics normalization")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    values = []
    for row in read_jsonl(args.manifest):
        cache = load_artifact(row["cache_path"])
        values.append(cache["physics"].float())
    matrix = torch.cat(values)
    center = matrix.median(0).values
    scale = (matrix - center).abs().median(0).values * 1.4826
    scale = torch.where(scale > 1e-6, scale, torch.ones_like(scale))
    atomic_save({"center": center, "scale": scale, "clips": len(values)}, args.output)
    print(f"saved {args.output}: {tuple(matrix.shape)}")


if __name__ == "__main__":
    main()
