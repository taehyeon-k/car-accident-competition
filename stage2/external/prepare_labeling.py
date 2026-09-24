"""Create browser-compatible proxies and a resumable manual-labeling queue."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
from typing import Any


DEFAULT_ROOT = Path(os.environ.get("STAGE2_EXTERNAL_ROOT", "/workspace/data/stage2_external"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def codec(path: Path) -> str:
    process = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name", "-of", "default=nw=1:nk=1", str(path)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    return process.stdout.strip()


def proxy(original: Path, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file():
        return output
    process = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y", "-i", str(original), "-map", "0:v:0", "-an",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-fps_mode", "passthrough", "-movflags", "+faststart", str(output),
        ],
        text=True, stderr=subprocess.PIPE,
    )
    if process.returncode:
        output.unlink(missing_ok=True)
        raise RuntimeError(process.stderr[-4000:])
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    arguments = parser.parse_args()
    source = arguments.root / "manifests/high_confidence_label_queue.jsonl"
    rows = [row for row in read_jsonl(source) if row.get("dedup_status") not in {"EXACT_DUPLICATE", "NEAR_DUPLICATE"}]
    labeling = arguments.root / "labeling"
    proxies = labeling / "proxies"
    output_rows = []
    for index, row in enumerate(rows, 1):
        original = Path(row["video_path"])
        video_codec = codec(original)
        playback = original
        if video_codec != "h264":
            print(f"[{index}/{len(rows)}] H.264 proxy for {row['sample_id']} ({video_codec})", flush=True)
            playback = proxy(original, proxies / f"{row['sample_id']}.mp4")
        mapping_path = original.with_suffix(".frames.json")
        if mapping_path.is_file():
            mapping = json.loads(mapping_path.read_text(encoding="utf-8"))["position_to_original_frame"]
        else:
            mapping = list(range(int(row["num_frames"])))
        candidate = int(row["metadata_collision_frame_candidate"])
        try:
            candidate_position = mapping.index(candidate)
        except ValueError:
            candidate_position = min(len(mapping) - 1, max(0, candidate))
        prepared = dict(row)
        prepared.update(
            original_video_path=str(original),
            playback_video_path=str(playback),
            position_to_original_frame=mapping,
            metadata_collision_position=candidate_position,
        )
        output_rows.append(prepared)
    labeling.mkdir(parents=True, exist_ok=True)
    with (labeling / "queue.jsonl").open("w", encoding="utf-8") as stream:
        for row in output_rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    if not (labeling / "annotations.jsonl").exists():
        (labeling / "annotations.jsonl").write_text("", encoding="utf-8")
    print(json.dumps({"queue": len(output_rows), "proxies": sum(Path(row["playback_video_path"]) != Path(row["original_video_path"]) for row in output_rows), "queue_path": str(labeling / "queue.jsonl"), "annotations_path": str(labeling / "annotations.jsonl")}, indent=2))


if __name__ == "__main__":
    main()
