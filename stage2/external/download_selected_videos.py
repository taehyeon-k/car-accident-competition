"""Download only metadata-selected source videos and maintain a complete ledger."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any


DEFAULT_ROOT = Path(os.environ.get("STAGE2_EXTERNAL_ROOT", "/workspace/data/stage2_external"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def disk_free_gib(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / 2**30


def classify_error(output: str) -> str:
    lowered = output.lower()
    if "private video" in lowered:
        return "PRIVATE"
    if "video unavailable" in lowered or "not available" in lowered or "removed" in lowered:
        return "REMOVED"
    return "DOWNLOAD_ERROR"


def download_causalcrash(row: dict[str, Any], output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    existing = sorted(output.glob(f"{row['source_id']}.*"))
    existing = [path for path in existing if path.suffix.lower() in {".mp4", ".webm", ".mkv", ".mov"}]
    if existing:
        path = existing[0]
        return {"download_status": "AVAILABLE_AND_DOWNLOADED", "local_path": str(path), "error": "", "bytes": path.stat().st_size, "checksum": sha256(path)}
    template = str(output / f"{row['source_id']}.%(ext)s")
    command = [
        "yt-dlp",
        "--no-playlist",
        "--no-progress",
        "--format",
        "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
        "--merge-output-format",
        "mp4",
        "--print",
        "after_move:filepath",
        "--output",
        template,
        row["source_url"],
    ]
    process = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output_text = process.stdout.strip()
    candidates = sorted(output.glob(f"{row['source_id']}.*"))
    candidates = [path for path in candidates if path.suffix.lower() in {".mp4", ".webm", ".mkv", ".mov"}]
    if process.returncode or not candidates:
        return {"download_status": classify_error(output_text), "local_path": "", "error": output_text[-4000:], "bytes": 0, "checksum": ""}
    path = candidates[0]
    return {"download_status": "AVAILABLE_AND_DOWNLOADED", "local_path": str(path), "error": "", "bytes": path.stat().st_size, "checksum": sha256(path)}


def write_ledger(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["source", "source_id", "source_url", "filter_decision", "remote_path", "shard", "download_status", "local_path", "bytes", "checksum", "error"]
    with path.with_suffix(".jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--source", choices=("causalcrash",), default="causalcrash")
    parser.add_argument("--minimum-free-gib", type=float, default=15.0)
    parser.add_argument("--limit", type=int)
    arguments = parser.parse_args()

    normalized = read_jsonl(arguments.root / f"metadata/{arguments.source}/normalized.jsonl")
    selected = [row for row in normalized if row["filter_decision"] == "HIGH_CONFIDENCE_LABEL_QUEUE"]
    if arguments.limit is not None:
        selected_ids = {row["source_id"] for row in selected[: arguments.limit]}
    else:
        selected_ids = {row["source_id"] for row in selected}
    output = arguments.root / "raw_selected" / arguments.source
    existing_ledger_path = arguments.root / "manifests" / f"{arguments.source}_download_ledger.jsonl"
    previous = {row["source_id"]: row for row in read_jsonl(existing_ledger_path)} if existing_ledger_path.exists() else {}
    ledger = []
    for index, row in enumerate(normalized, 1):
        record = {
            "source": arguments.source,
            "source_id": row["source_id"],
            "source_url": row["source_url"],
            "filter_decision": row["filter_decision"],
            "remote_path": row["source_url"],
            "shard": row.get("mmau_archive", ""),
            "download_status": "NOT_SELECTED",
            "local_path": "",
            "bytes": 0,
            "checksum": "",
            "error": "",
        }
        if row["source_id"] in selected_ids:
            if disk_free_gib(arguments.root) < arguments.minimum_free_gib:
                record.update(download_status="DOWNLOAD_ERROR", error=f"free space below {arguments.minimum_free_gib:.1f} GiB safety floor")
            else:
                print(f"[{index}/{len(normalized)}] downloading {row['source_id']}", flush=True)
                record.update(download_causalcrash(row, output))
                print(f"  {record['download_status']}", flush=True)
        elif row["source_id"] in previous and previous[row["source_id"]].get("download_status") == "AVAILABLE_AND_DOWNLOADED":
            record.update(previous[row["source_id"]])
        ledger.append(record)
        write_ledger(arguments.root / "manifests" / f"{arguments.source}_download_ledger", ledger)
    failures = [row for row in ledger if row["filter_decision"] == "HIGH_CONFIDENCE_LABEL_QUEUE" and row["download_status"] != "AVAILABLE_AND_DOWNLOADED"]
    write_ledger(arguments.root / "manifests/download_failures", failures)
    print(json.dumps({"selected": len(selected_ids), "downloaded": sum(row["download_status"] == "AVAILABLE_AND_DOWNLOADED" for row in ledger), "failed": len(failures), "free_gib": round(disk_free_gib(arguments.root), 2)}, indent=2))


if __name__ == "__main__":
    main()
