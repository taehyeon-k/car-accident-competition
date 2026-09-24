"""Probe and fully decode downloaded candidates, then build the valid label queue."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any


DEFAULT_ROOT = Path(os.environ.get("STAGE2_EXTERNAL_ROOT", "/workspace/data/stage2_external"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def ratio(value: str | None) -> float:
    if not value or value == "0/0":
        return 0.0
    numerator, denominator = value.split("/", 1)
    return float(numerator) / float(denominator)


def probe(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames,time_base,start_time,duration:format=duration,start_time",
        "-of", "json", str(path),
    ]
    process = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode:
        return {"validation_status": "INVALID", "error": process.stderr.strip(), "decodes": False}
    data = json.loads(process.stdout)
    if not data.get("streams"):
        return {"validation_status": "INVALID", "error": "no video stream", "decodes": False}
    stream = data["streams"][0]
    fmt = data.get("format", {})
    fps = ratio(stream.get("avg_frame_rate")) or ratio(stream.get("r_frame_rate"))
    duration = float(stream.get("duration") or fmt.get("duration") or 0)
    frames = int(stream.get("nb_read_frames") or stream.get("nb_frames") or round(duration * fps))
    decode = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"],
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    valid = decode.returncode == 0 and duration > 0 and fps > 0 and frames > 0
    return {
        "validation_status": "VALID" if valid else "INVALID",
        "error": decode.stderr.strip()[-4000:] if decode.returncode else "",
        "decodes": decode.returncode == 0,
        "codec": stream.get("codec_name", ""),
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "duration_s": duration,
        "average_fps": fps,
        "nominal_fps": ratio(stream.get("r_frame_rate")),
        "num_frames": frames,
        "time_base": stream.get("time_base", ""),
        "start_time": float(stream.get("start_time") or fmt.get("start_time") or 0),
    }


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(row.keys() for row in rows))) if rows else ["source", "source_id", "download_status", "error"]
    with path.with_suffix(".jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    arguments = parser.parse_args()
    manifests = arguments.root / "manifests"
    ledgers = []
    for ledger_path in sorted(manifests.glob("*_download_ledger.jsonl")):
        ledgers.extend(read_jsonl(ledger_path))
    failures = [
        row
        for row in ledgers
        if row.get("filter_decision") == "HIGH_CONFIDENCE_LABEL_QUEUE"
        and row.get("download_status") not in {"AVAILABLE_AND_DOWNLOADED", "NOT_SELECTED_ARCHIVE_LIMIT"}
    ]
    archive_limited = [row for row in ledgers if row.get("download_status") == "NOT_SELECTED_ARCHIVE_LIMIT"]
    write_table(manifests / "download_failures", failures)
    write_table(manifests / "archive_not_fetched", archive_limited)
    downloaded = [row for row in ledgers if row.get("download_status") == "AVAILABLE_AND_DOWNLOADED"]
    normalized = {}
    for source in ("mmau", "causalcrash"):
        path = arguments.root / f"metadata/{source}/normalized.jsonl"
        if path.exists():
            normalized.update({(row["source"], row["source_id"]): row for row in read_jsonl(path)})
    validation = []
    label_queue = []
    for index, download in enumerate(downloaded, 1):
        path = Path(download["local_path"])
        print(f"[{index}/{len(downloaded)}] validating {path.name}", flush=True)
        result = {"source": download["source"], "source_id": download["source_id"], "local_path": str(path)}
        result.update(probe(path))
        metadata = normalized[(download["source"], download["source_id"])]
        collision_time = metadata.get("metadata_collision_time")
        if isinstance(collision_time, (int, float)) and result.get("average_fps"):
            result["metadata_collision_inside"] = 0 <= collision_time <= result["duration_s"]
            result["metadata_collision_frame_candidate"] = min(
                result["num_frames"] - 1,
                max(0, round(collision_time * result["average_fps"])),
            )
        else:
            candidate = metadata.get("metadata_collision_frame_candidate")
            frame_map_path = path.with_suffix(".frames.json")
            if frame_map_path.is_file():
                frame_map = json.loads(frame_map_path.read_text(encoding="utf-8")).get("position_to_original_frame", [])
                result["frame_map_count"] = len(frame_map)
                result["metadata_collision_inside"] = isinstance(candidate, int) and candidate in frame_map
                if result["metadata_collision_inside"]:
                    result["metadata_collision_position"] = frame_map.index(candidate)
            else:
                result["metadata_collision_inside"] = isinstance(candidate, int) and 0 <= candidate < result.get("num_frames", 0)
            result["metadata_collision_frame_candidate"] = candidate
        if not result["metadata_collision_inside"]:
            result["validation_status"] = "INVALID"
            result["error"] = (result.get("error", "") + " collision metadata outside decoded video").strip()
        if result.get("frame_map_count") is not None and result["frame_map_count"] != result.get("num_frames"):
            result["validation_status"] = "INVALID"
            result["error"] = (
                result.get("error", "")
                + f" frame map has {result['frame_map_count']} entries but video decodes to {result.get('num_frames')} frames"
            ).strip()
        validation.append(result)
        if result["validation_status"] == "VALID" and metadata["filter_decision"] == "HIGH_CONFIDENCE_LABEL_QUEUE":
            queue_row = dict(metadata)
            queue_row.update(
                video_path=str(path),
                fps=result["average_fps"],
                num_frames=result["num_frames"],
                duration_s=result["duration_s"],
                width=result["width"],
                height=result["height"],
                metadata_collision_frame_candidate=result["metadata_collision_frame_candidate"],
                validation_status="VALID",
            )
            label_queue.append(queue_row)
    write_table(manifests / "video_validation", validation)
    write_table(manifests / "high_confidence_label_queue", label_queue)
    print(json.dumps({"downloaded": len(downloaded), "valid": sum(row["validation_status"] == "VALID" for row in validation), "label_queue": len(label_queue)}, indent=2))


if __name__ == "__main__":
    main()
