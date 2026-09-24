"""Find exact and sampled-frame near duplicates without extracting full videos."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import cv2
import numpy as np


DEFAULT_ROOT = Path(os.environ.get("STAGE2_EXTERNAL_ROOT", "/workspace/data/stage2_external"))
CONFIDENT_MATCHED = 6
CONFIDENT_MEDIAN_BITS = 3.0
EXISTING_STAGE2 = Path("/workspace/data/stage2/usable/videos")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def duration(path: Path) -> float:
    process = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    try:
        return float(process.stdout.strip())
    except ValueError:
        return 0.0


def perceptual_hashes(path: Path, samples: int = 7) -> list[int]:
    seconds = duration(path)
    if seconds <= 0:
        return []
    rate = samples / seconds
    process = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(path), "-vf", f"fps={rate:.12f},scale=32:32,format=gray",
            "-frames:v", str(samples), "-f", "rawvideo", "-pix_fmt", "gray", "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    size = 32 * 32
    frames = [process.stdout[offset : offset + size] for offset in range(0, len(process.stdout), size)]
    hashes = []
    for raw in frames[:samples]:
        if len(raw) != size:
            continue
        image = np.frombuffer(raw, dtype=np.uint8).reshape(32, 32).astype(np.float32)
        low = cv2.dct(image)[:8, :8]
        threshold = np.median(low[1:])
        bits = (low > threshold).reshape(-1)
        value = 0
        for bit in bits:
            value = (value << 1) | int(bit)
        hashes.append(value)
    return hashes


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def near_score(left: list[int], right: list[int]) -> tuple[int, float]:
    if not left or not right:
        return 0, 64.0
    distances = [min(hamming(item, candidate) for candidate in right) for item in left]
    return sum(distance <= 8 for distance in distances), float(np.median(distances))


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with path.with_suffix(".jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--existing-stage2", type=Path, default=EXISTING_STAGE2)
    arguments = parser.parse_args()
    queue_path = arguments.root / "manifests/high_confidence_label_queue.jsonl"
    queue = read_jsonl(queue_path)
    candidates = [(row["sample_id"], Path(row["video_path"]), "candidate") for row in queue]
    existing = [(f"existing:{path.stem}", path, "existing_stage2") for path in sorted(arguments.existing_stage2.glob("*.mp4"))]
    items = candidates + existing
    fingerprints = {}
    exact = {}
    for index, (sample_id, path, group) in enumerate(items, 1):
        print(f"[{index}/{len(items)}] fingerprinting {sample_id}", flush=True)
        checksum = sha256(path)
        hashes = perceptual_hashes(path)
        fingerprints[sample_id] = hashes
        exact.setdefault(checksum, []).append(sample_id)
    reports = []
    source_references: dict[tuple[str, str], list[str]] = {}
    urls: dict[str, list[str]] = {}
    for row in queue:
        source_references.setdefault((row["source"], row["source_id"]), []).append(row["sample_id"])
        url = str(row.get("source_url", "")).strip()
        if url and "huggingface.co/datasets/JeffreyChou/MM-AU" not in url:
            urls.setdefault(url, []).append(row["sample_id"])
    reference_duplicate_candidates: set[str] = set()
    for reference, members in [*source_references.items(), *urls.items()]:
        if len(members) < 2:
            continue
        keeper, *rejected = members
        reference_duplicate_candidates.update(rejected)
        for duplicate in rejected:
            reports.append(
                {
                    "kind": "DUPLICATE_SOURCE_REFERENCE",
                    "left": duplicate,
                    "right": keeper,
                    "evidence": str(reference),
                    "action": "REJECT_DUPLICATE",
                }
            )
    exact_duplicate_candidates: set[str] = set()
    for checksum, members in exact.items():
        candidates_in_group = [member for member in members if not member.startswith("existing:")]
        existing_in_group = [member for member in members if member.startswith("existing:")]
        if not candidates_in_group or len(members) < 2:
            continue
        if existing_in_group:
            rejected = candidates_in_group
            keeper = existing_in_group[0]
        else:
            keeper, *rejected = candidates_in_group
        exact_duplicate_candidates.update(rejected)
        for duplicate in rejected:
            reports.append(
                {
                    "kind": "EXACT_FILE",
                    "left": duplicate,
                    "right": keeper,
                    "evidence": checksum,
                    "action": "REJECT_DUPLICATE",
                }
            )
    near_duplicate_candidates: set[str] = set()
    for i, (left_id, _, _) in enumerate(candidates):
        for right_id, _, right_group in items[i + 1 :]:
            matched, median = near_score(fingerprints[left_id], fingerprints[right_id])
            evidence = f"{matched}/7 sampled pHashes within 8 bits; median={median:.1f}"
            if matched >= CONFIDENT_MATCHED and median <= CONFIDENT_MEDIAN_BITS:
                # Visually verified on sample pairs: these are the same source
                # footage re-trimmed.  Keep the earlier candidate (or the
                # existing Stage-2 clip) and drop the other from the queue.
                duplicate = left_id if right_group == "existing_stage2" else right_id
                keeper = right_id if duplicate == left_id else left_id
                near_duplicate_candidates.add(duplicate)
                reports.append({"kind": "CONFIDENT_NEAR_DUPLICATE", "left": duplicate, "right": keeper, "evidence": evidence, "action": "REJECT_DUPLICATE", "right_group": right_group})
            elif matched >= 4 and median <= 8:
                reports.append({"kind": "POTENTIAL_NEAR_DUPLICATE", "left": left_id, "right": right_id, "evidence": evidence, "action": "REPORT_ONLY", "right_group": right_group})

    mmau = read_jsonl(arguments.root / "metadata/mmau/normalized.jsonl")
    by_name = {}
    for row in mmau:
        by_name.setdefault(row["mmau_video_name"], []).append(row["sample_id"])
    for name, members in by_name.items():
        if len(members) > 1:
            reports.append({"kind": "DUPLICATE_RELEASE_VIDEO_NAME", "left": members[0], "right": members[1], "evidence": name, "action": "REPORT_ONLY_UNTIL_PIXELS_AVAILABLE"})
    status = {row["sample_id"]: "UNIQUE" for row in queue}
    for sample_id in exact_duplicate_candidates | reference_duplicate_candidates:
        status[sample_id] = "EXACT_DUPLICATE"
    for sample_id in near_duplicate_candidates - exact_duplicate_candidates - reference_duplicate_candidates:
        status[sample_id] = "NEAR_DUPLICATE"
    for report in reports:
        if report["kind"] == "POTENTIAL_NEAR_DUPLICATE":
            if report["left"] in status and status[report["left"]] == "UNIQUE":
                status[report["left"]] = "POTENTIAL_NEAR_DUPLICATE"
            if report["right"] in status and status[report["right"]] == "UNIQUE":
                status[report["right"]] = "POTENTIAL_NEAR_DUPLICATE"
    for row in queue:
        row["dedup_status"] = status[row["sample_id"]]
    write_table(arguments.root / "manifests/dedup_report", reports)
    write_table(arguments.root / "manifests/high_confidence_label_queue", queue)
    (arguments.root / "metadata/video_fingerprints.json").write_text(
        json.dumps({sample_id: [f"{value:016x}" for value in values] for sample_id, values in fingerprints.items()}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"candidates": len(candidates), "existing": len(existing), "reports": len(reports), "exact": sum(row["kind"] == "EXACT_FILE" for row in reports), "source_reference": sum(row["kind"] == "DUPLICATE_SOURCE_REFERENCE" for row in reports), "confident_near": sum(row["kind"] == "CONFIDENT_NEAR_DUPLICATE" for row in reports), "potential": sum(row["kind"] == "POTENTIAL_NEAR_DUPLICATE" for row in reports)}, indent=2))


if __name__ == "__main__":
    main()
