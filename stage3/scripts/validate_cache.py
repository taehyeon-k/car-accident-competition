from __future__ import annotations

import argparse

import torch

from stage3.utils.config import read_jsonl
from stage3.data.cache import dequantize_motion
from stage3.utils.checkpoint import load_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Stage 3 cache shapes and finite values")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    rows = read_jsonl(args.manifest)
    for row in rows[: args.limit]:
        value = load_artifact(row["cache_path"])
        motion, physics = dequantize_motion(value), value["physics"]
        if motion.ndim != 4 or motion.shape[1:] != (10, 96, 168):
            raise ValueError(f"Bad motion shape for {row['clip_id']}: {tuple(motion.shape)}")
        if physics.shape != (len(motion), 20) or not torch.isfinite(motion).all() or not torch.isfinite(physics).all():
            raise ValueError(f"Invalid physics/features for {row['clip_id']}")
        print(row["clip_id"], tuple(motion.shape), value["cache_key"])


if __name__ == "__main__":
    main()
