from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from stage3.data.adapters.baton import BatonAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Build route-disjoint BATON Stage 3 manifests")
    parser.add_argument("--data-root", default="/workspace/data/stage3/BATON-Sample")
    parser.add_argument("--output-dir", default="/workspace/data/stage3/manifests")
    parser.add_argument("--cache-dir", default="/workspace/cache/stage3/motion")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--segment-seconds", type=float, default=30.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    records = BatonAdapter(args.data_root).discover()
    if not 0 < args.validation_fraction < 1:
        raise ValueError("validation-fraction must be between zero and one")
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(records))
    validation_routes = {records[i].clip_id for i in order[: max(1, round(len(records) * args.validation_fraction))]}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    partitions = {"train": [], "val": [], "all": []}
    for record in records:
        duration = float(record.signals.t[-1])
        for segment_index, start in enumerate(np.arange(0.0, duration, args.segment_seconds)):
            end = min(float(start + args.segment_seconds), duration)
            if end - start < 2.0:
                continue
            clip_id = f"{record.clip_id}__{segment_index:05d}"
            metadata = {**record.metadata, "source_clip_id": record.clip_id, "segment_start_time": float(start), "segment_end_time": end}
            row = {
                "clip_id": clip_id, "video_path": str(record.video_path),
                "signals_path": record.metadata["signals_path"],
                "cache_path": str(Path(args.cache_dir) / f"{clip_id}.pt"),
                "group_keys": record.group_keys, "metadata": metadata,
            }
            partitions["all"].append(row)
            partitions["val" if record.clip_id in validation_routes else "train"].append(row)
    smoke_dir = Path(args.cache_dir).parent / "smoke"
    partitions["smoke_train"] = [{**partitions["train"][0], "cache_path": str(smoke_dir / Path(partitions["train"][0]["cache_path"]).name)}]
    partitions["smoke_val"] = [{**partitions["val"][0], "cache_path": str(smoke_dir / Path(partitions["val"][0]["cache_path"]).name)}]
    for name, rows in partitions.items():
        path = output / f"{name}.jsonl"
        if path.exists() and not args.force:
            raise FileExistsError(f"{path} exists; use --force to replace it")
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    print({name: len(rows) for name, rows in partitions.items()})


if __name__ == "__main__":
    main()
