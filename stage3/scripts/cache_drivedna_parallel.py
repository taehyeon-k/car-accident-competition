"""Resume selected DriveDNA motion caching in bounded GPU worker processes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from stage3.utils.config import read_jsonl


def _gpu_usage_mb(pids: set[int]) -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    total = 0
    for line in result.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 2 and int(parts[0]) in pids:
            total += int(parts[1])
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--filter-report", type=Path, required=True)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-vram-mb", type=int, default=19000)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4 or not 1000 <= args.max_vram_mb < 20000:
        parser.error("Require 1-4 workers and max-vram-mb in [1000, 20000)")
    rows = read_jsonl(args.manifest)
    if any(row["metadata"].get("source") != "drivedna" for row in rows):
        parser.error("Manifest must contain only DriveDNA rows")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    marker = args.work_dir / "complete.json"
    if marker.is_file() and all(Path(row["cache_path"]).is_file() for row in rows):
        print(f"already_complete={len(rows)}", flush=True)
        return
    children, streams = [], []
    try:
        for i in range(args.workers):
            shard = args.work_dir / f"shard_{i}.jsonl"
            shard.write_text("".join(json.dumps(row) + "\n" for row in rows[i::args.workers]))
            log = open(args.work_dir / f"worker_{i}.log", "a", encoding="utf-8")
            streams.append(log)
            cmd = [sys.executable, "-m", "stage3.scripts.cache_motion",
                   "--config", str(args.config), "--manifest", str(shard),
                   "--source", "drivedna", "--data-root", str(args.raw_dir),
                   "--filter-report", str(args.filter_report),
                   "--selection-report", str(args.selection_report),
                   "--device", "cuda", "--gpu-memory-fraction", "0.12"]
            children.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT))
        pids = {child.pid for child in children}
        print(f"workers={sorted(pids)} segments={len(rows)}", flush=True)
        while True:
            running = [child for child in children if child.poll() is None]
            usage = _gpu_usage_mb(pids)
            if usage > args.max_vram_mb:
                raise RuntimeError(f"DriveDNA caching exceeded GPU cap: {usage} MiB")
            if not running:
                break
            done = sum(Path(row["cache_path"]).is_file() for row in rows)
            print(f"cached={done}/{len(rows)} gpu_vram_mb={usage}", flush=True)
            time.sleep(30)
        codes = [child.returncode for child in children]
        if any(code != 0 for code in codes):
            raise RuntimeError(f"Cache worker failures: {codes}; see {args.work_dir}/worker_*.log")
        missing = [row["cache_path"] for row in rows if not Path(row["cache_path"]).is_file()]
        if missing:
            raise RuntimeError(f"Missing {len(missing)} caches; first: {missing[0]}")
        marker.write_text(json.dumps({"manifest": str(args.manifest), "segments": len(rows)}) + "\n")
        print(f"complete={len(rows)}", flush=True)
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            child.wait()
        for stream in streams:
            stream.close()


if __name__ == "__main__":
    main()
