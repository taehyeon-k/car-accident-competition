"""Stream one MM-AU split archive and retain only selected frame sequences."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

from huggingface_hub import HfApi, hf_hub_url


DEFAULT_ROOT = Path(os.environ.get("STAGE2_EXTERNAL_ROOT", "/workspace/data/stage2_external"))
REPO = "JeffreyChou/MM-AU"
REVISION = "540cb1277cb70e91a7022abe852decb3ee9adb0a"
FPS = 30.0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parts(archive: str) -> list[tuple[str, int]]:
    info = HfApi().dataset_info(REPO, revision=REVISION, files_metadata=True)
    prefix = archive.rstrip("/") + "/"
    return sorted((item.rfilename, int(item.size or 0)) for item in info.siblings if item.rfilename.startswith(prefix))


def stream_extract(archive_parts: list[tuple[str, int]], patterns: list[str], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as pattern_file:
        for pattern in patterns:
            pattern_file.write(pattern.rstrip("/") + "/*\n")
        pattern_path = Path(pattern_file.name)
    tar = subprocess.Popen(
        ["tar", "-xzf", "-", "-C", str(destination), "--wildcards", "--files-from", str(pattern_path)],
        stdin=subprocess.PIPE,
    )
    assert tar.stdin is not None
    total = sum(size for _, size in archive_parts)
    transferred = 0
    try:
        for index, (filename, size) in enumerate(archive_parts, 1):
            url = hf_hub_url(REPO, filename, repo_type="dataset", revision=REVISION)
            print(f"part {index}/{len(archive_parts)}: {filename} ({size / 2**30:.2f} GiB)", flush=True)
            curl = subprocess.Popen(
                ["curl", "-fL", "--retry", "5", "--retry-all-errors", "--silent", "--show-error", url],
                stdout=subprocess.PIPE,
            )
            assert curl.stdout is not None
            for chunk in iter(lambda: curl.stdout.read(8 * 1024 * 1024), b""):
                tar.stdin.write(chunk)
                transferred += len(chunk)
            curl.stdout.close()
            if curl.wait() != 0:
                raise RuntimeError(f"download failed for {filename}")
            print(f"  streamed {transferred / 2**30:.2f}/{total / 2**30:.2f} GiB", flush=True)
        tar.stdin.close()
        if tar.wait() != 0:
            raise RuntimeError("tar extraction failed")
    except Exception:
        tar.kill()
        raise
    finally:
        pattern_path.unlink(missing_ok=True)


def make_video(images: Path, output: Path, frame_map: Path) -> int:
    paths = sorted(path for path in images.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not paths:
        raise ValueError(f"No frames extracted in {images}")
    ids = [int(path.stem) for path in paths]
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_map.write_text(json.dumps({"fps": FPS, "position_to_original_frame": ids}, indent=2) + "\n", encoding="utf-8")
    suffixes = {path.suffix.lower() for path in paths}
    if len(suffixes) != 1:
        raise ValueError(f"Mixed frame extensions in {images}: {sorted(suffixes)}")
    pattern = str(images / f"*{paths[0].suffix}")
    output.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-pattern_type", "glob", "-i", pattern,
            "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(output),
        ],
        text=True,
        stderr=subprocess.PIPE,
    )
    if process.returncode:
        output.unlink(missing_ok=True)
        raise RuntimeError(process.stderr[-4000:])
    return len(paths)


def write_ledger(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["source", "source_id", "source_url", "filter_decision", "remote_path", "shard", "download_status", "local_path", "bytes", "checksum", "error"]
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
    parser.add_argument("--archive", default="CAP-DATA_chunks/1-10")
    parser.add_argument("--minimum-free-gib", type=float, default=20.0)
    parser.add_argument(
        "--reuse-extracted", action="store_true",
        help="skip streaming; encode sequences already present under raw_selected/mmau_frames",
    )
    arguments = parser.parse_args()
    free = shutil.disk_usage(arguments.root).free / 2**30
    if free < arguments.minimum_free_gib:
        raise RuntimeError(f"Only {free:.2f} GiB free; safety floor is {arguments.minimum_free_gib:.2f} GiB")
    normalized = read_jsonl(arguments.root / "metadata/mmau/normalized.jsonl")
    selected = [
        row for row in normalized
        if row["filter_decision"] == "HIGH_CONFIDENCE_LABEL_QUEUE" and row["mmau_archive"] == arguments.archive
    ]
    if not selected:
        raise ValueError(f"No selected records in {arguments.archive}")
    temporary = arguments.root / "tmp/mmau_extract"
    raw_frames = arguments.root / "raw_selected/mmau_frames"
    if not arguments.reuse_extracted:
        archive_parts = parts(arguments.archive)
        print(json.dumps({"archive": arguments.archive, "parts": len(archive_parts), "compressed_gib": round(sum(size for _, size in archive_parts) / 2**30, 2), "selected_samples": len(selected), "free_gib": round(free, 2)}, indent=2), flush=True)
        stream_extract(archive_parts, [row["mmau_member_prefix"] for row in selected], temporary)
    videos = arguments.root / "raw_selected/mmau"
    outcomes = {}
    for index, row in enumerate(selected, 1):
        extracted = temporary / row["mmau_member_prefix"]
        destination = raw_frames / row["source_id"]
        try:
            if extracted.exists():
                if destination.exists():
                    shutil.rmtree(destination)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(extracted), str(destination))
            elif not destination.exists():
                raise FileNotFoundError(f"No extracted frames for {row['source_id']}")
            video = videos / f"{row['source_id']}.mp4"
            frame_map = videos / f"{row['source_id']}.frames.json"
            count = make_video(destination, video, frame_map)
            outcomes[row["source_id"]] = {
                "download_status": "AVAILABLE_AND_DOWNLOADED",
                "local_path": str(video),
                "bytes": video.stat().st_size,
                "checksum": sha256(video),
                "error": "" if count == row["num_frames"] else f"metadata has {row['num_frames']} frames; archive has {count}",
            }
            # The release contains JPEG sequences rather than source videos.  The
            # MP4 and sidecar preserve every frame and its original numeric ID;
            # retaining a second permanent frame tree would waste local/R2 space.
            shutil.rmtree(destination)
        except Exception as error:
            outcomes[row["source_id"]] = {"download_status": "DOWNLOAD_ERROR", "local_path": "", "bytes": 0, "checksum": "", "error": str(error)}
        print(f"[{index}/{len(selected)}] {row['source_id']}: {outcomes[row['source_id']]['download_status']}", flush=True)

    ledger_path = arguments.root / "manifests/mmau_download_ledger.jsonl"
    previous = {}
    if ledger_path.exists():
        previous = {
            row["source_id"]: row for row in read_jsonl(ledger_path)
            if row["download_status"] == "AVAILABLE_AND_DOWNLOADED"
        }
    ledger = []
    for row in normalized:
        record = {
            "source": "mmau", "source_id": row["source_id"], "source_url": row["source_url"],
            "filter_decision": row["filter_decision"], "remote_path": row["mmau_member_prefix"],
            "shard": row["mmau_archive"], "download_status": "NOT_SELECTED", "local_path": "",
            "bytes": 0, "checksum": "", "error": "",
        }
        if row["source_id"] in outcomes:
            record.update(outcomes[row["source_id"]])
        elif row["source_id"] in previous:
            record = previous[row["source_id"]]
        elif row["filter_decision"] == "HIGH_CONFIDENCE_LABEL_QUEUE":
            record.update(download_status="NOT_SELECTED_ARCHIVE_LIMIT", error="archive not streamed in this run")
        ledger.append(record)
    write_ledger(arguments.root / "manifests/mmau_download_ledger", ledger)
    try:
        raw_frames.rmdir()
    except OSError:
        pass
    shutil.rmtree(temporary, ignore_errors=True)
    print(json.dumps({"selected": len(selected), "downloaded": sum(value["download_status"] == "AVAILABLE_AND_DOWNLOADED" for value in outcomes.values()), "failed": sum(value["download_status"] != "AVAILABLE_AND_DOWNLOADED" for value in outcomes.values()), "free_gib": round(shutil.disk_usage(arguments.root).free / 2**30, 2)}, indent=2))


if __name__ == "__main__":
    main()
