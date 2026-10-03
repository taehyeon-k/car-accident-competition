from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from stage3.data.adapters.baton import BatonAdapter
from stage3.data.cache import cache_record, cache_key
from stage3.utils.checkpoint import load_artifact
from stage3.flow import build_flow_estimator
from stage3.utils.config import load_config, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache versioned Stage 3 motion features")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", default="/workspace/data/stage3/BATON-Sample")
    parser.add_argument("--device")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    available = {record.clip_id: record for record in BatonAdapter(args.data_root).discover()}
    rows = read_jsonl(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    estimator = build_flow_estimator(cfg["flow"], args.device)
    for row in rows:
        source_id = row.get("metadata", {}).get("source_clip_id", row["clip_id"])
        if source_id not in available:
            raise KeyError(f"Manifest route not found in BATON: {source_id}")
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
