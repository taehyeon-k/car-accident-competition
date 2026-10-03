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
    cache_keys = set()
    for row in read_jsonl(args.manifest):
        cache = load_artifact(row["cache_path"])
        cache_keys.add(cache["cache_key"])
        values.append(cache["physics"][cache["time_valid"].bool()].float())
    if len(cache_keys) != 1:
        raise ValueError("Training caches use mixed feature versions; rebuild them first")
    matrix = torch.cat(values)
    if not len(matrix):
        raise ValueError("No valid training frames for normalization")
    center = matrix.median(0).values
    scale = (matrix - center).abs().median(0).values * 1.4826
    scale = torch.where(scale > 1e-6, scale, torch.ones_like(scale))
    atomic_save({"center": center, "scale": scale, "clips": len(values), "cache_key": next(iter(cache_keys))}, args.output)
    print(f"saved {args.output}: {tuple(matrix.shape)}")


if __name__ == "__main__":
    main()
