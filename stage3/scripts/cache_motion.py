from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from stage3.data.adapters.baton import BatonAdapter
from stage3.data.adapters.drivedna import DriveDnaAdapter
from stage3.data.cache import cache_record, cache_key
from stage3.utils.checkpoint import load_artifact
from stage3.flow import build_flow_estimator
from stage3.utils.config import load_config, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache versioned Stage 3 motion features")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", default="/workspace/data/stage3/BATON-Sample")
    parser.add_argument("--source", choices=("baton", "drivedna"), default="baton")
    parser.add_argument("--filter-report", help="required when --source drivedna")
    parser.add_argument("--selection-report", help="deduplicated DriveDNA selection report")
    parser.add_argument("--device")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--gpu-memory-fraction", type=float,
                        help="cap this process's PyTorch allocation as a fraction of device VRAM")
    args = parser.parse_args()
    if args.gpu_memory_fraction is not None:
        if not 0 < args.gpu_memory_fraction <= 1:
            parser.error("--gpu-memory-fraction must be in (0, 1]")
        if not torch.cuda.is_available():
            parser.error("CUDA is required for --gpu-memory-fraction")
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, 0)
    cfg = load_config(args.config)
    if args.source == "drivedna":
        if not args.filter_report:
            parser.error("--filter-report is required when --source drivedna")
        adapter = DriveDnaAdapter(args.data_root, args.filter_report, args.selection_report)
    else:
        adapter = BatonAdapter(args.data_root)
    available = {record.clip_id: record for record in adapter.discover()}
    rows = read_jsonl(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    estimator = build_flow_estimator(cfg["flow"], args.device)
    for row in rows:
        source_id = row.get("metadata", {}).get("source_clip_id", row["clip_id"])
        if source_id not in available:
            raise KeyError(f"Manifest source drive not found in {args.source}: {source_id}")
        record = replace(available[source_id], clip_id=row["clip_id"], metadata=row.get("metadata", available[source_id].metadata))
        output = Path(row["cache_path"])
        if output.exists() and not args.force:
            if load_artifact(output).get("cache_key") == cache_key(cfg):
                print(f"skip {output}")
                continue
            print(f"rebuild stale cache {output}")
        report = cache_record(record, cfg, output, args.max_frames, args.device, estimator)
        print(report)


if __name__ == "__main__":
    main()
