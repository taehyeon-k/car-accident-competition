"""Convert the workspace CSV/videos to native zero-based frames and source-safe JSONL splits."""

import argparse
import csv
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from stage2.scripts.prepare_manifest import split_rows


def extract(row):
    directory = Path(row["frames_dir"])
    expected = row["num_frames"]
    existing = sorted(directory.glob("*.jpg"))
    if len(existing) == expected and all(
        p.stem == f"{i:06d}" for i, p in enumerate(existing)
    ):
        return
    if existing:
        raise ValueError(
            f"Incomplete frame directory: {directory}; inspect before retrying"
        )
    directory.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-threads",
            "1",
            "-i",
            row["video_path"],
            "-fps_mode",
            "passthrough",
            "-start_number",
            "0",
            "-q:v",
            "3",
            "-threads",
            "1",
            str(directory / "%06d.jpg"),
        ],
        check=True,
    )
    actual = len(list(directory.glob("*.jpg")))
    if actual != expected:
        raise ValueError(
            f"{row['sample_id']}: decoded {actual}, CSV expected {expected}"
        )
    print(f"Extracted {row['sample_id']}: {actual} frames", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers and limit must be positive")
    args.workspace = args.workspace.resolve()
    base = args.workspace / "data/stage2"
    with (base / "usable/usable_only.csv").open() as f:
        records = list(csv.DictReader(f))
    rows = []
    for r in records:
        row = dict(
            sample_id=r["sample_id"],
            source_id=r["source"] + ":" + r["source_id"],
            video_path=str(base / r["video_relative_path"]),
            frames_dir=str(base / "frames" / r["sample_id"]),
            geometry_dir=str(args.workspace / "cache/geometry" / r["sample_id"]),
            native_fps=float(r["fps"]),
            num_frames=int(r["num_frames"]),
            entry_frame=int(r["entry_frame"]),
            collision_frame=int(r["collision_frame"]),
            entry_side=r["entry_side"],
            evasion_space=int(r["evasion_space"]),
        )
        if not Path(row["video_path"]).is_file():
            raise FileNotFoundError(row["video_path"])
        if not 0 <= row["entry_frame"] <= row["collision_frame"] < row["num_frames"]:
            raise ValueError(f"Invalid event bounds: {row['sample_id']}")
        if row["entry_side"] not in {"LEFT", "RIGHT"} or row["evasion_space"] not in {
            0,
            1,
        }:
            raise ValueError(f"Invalid categorical labels: {row['sample_id']}")
        rows.append(row)
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate sample IDs in CSV")
    train, val = split_rows(rows, 0.2, 42)
    output = base / "manifests"
    output.mkdir(exist_ok=True)
    for name, partition in [
        ("all", rows),
        ("train", train),
        ("val", val),
        ("smoke_train", train[:1]),
        ("smoke_val", val[:1]),
    ]:
        path = output / f"{name}.jsonl"
        content = "".join(json.dumps(row) + "\n" for row in partition)
        if path.exists() and path.read_text() != content:
            raise ValueError(f"Refusing to replace a different split: {path}")
        path.write_text(content)
    if args.extract:
        smoke = train[:1] + val[:1]
        smoke_ids = {row["sample_id"] for row in smoke}
        ordered = smoke + [row for row in rows if row["sample_id"] not in smoke_ids]
        selected = rows if args.limit is None else ordered[: args.limit]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(extract, selected))
    print(f"Manifests: {len(train)} train, {len(val)} validation; {output}")


if __name__ == "__main__":
    main()
