"""Record ADAS-TO/DriveDNA sample metadata before downloading any video.

Usage::

    python -m stage3.scripts.external_data_inventory --source adas \
        --output /workspace/data/stage3/adas_to/metadata
    python -m stage3.scripts.external_data_inventory --source drivedna \
        --output /workspace/data/stage3/drivedna/metadata

The inventory is useful even without gated-file access. Metadata and telemetry
transfers are explicit steps; this script never downloads video.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


REPOS = {"adas": "HenryYHW/ADAS-TO-Sample", "drivedna": "HenryYHW/DriveDNA-Sample"}


def inventory(source: str, output: Path, fetch_metadata: bool = False, fetch_telemetry: bool = False,
              raw_dir: Path | None = None) -> dict:
    if source not in REPOS:
        raise ValueError(f"Unknown source {source!r}; expected one of {sorted(REPOS)}")
    repo_id = REPOS[source]
    output.mkdir(parents=True, exist_ok=True)
    info = HfApi().repo_info(repo_id, repo_type="dataset", files_metadata=True)
    files = {s.rfilename: s.size for s in info.siblings or []}
    if source == "drivedna":
        csv = {name[:-4] for name in files if name.startswith("Dataset/") and name.endswith(".csv")}
        video = {name[:-4] for name in files if name.startswith("Dataset/") and name.endswith(".mp4")}
        if csv != video:
            raise ValueError(f"DriveDNA video/CSV mismatch: CSV-only={sorted(csv-video)}, video-only={sorted(video-csv)}")
        metadata_files = ["index.csv"]
        telemetry_files = sorted(name + ".csv" for name in csv)
        video_files = sorted(name + ".mp4" for name in video)
    else:
        video = {name.rsplit("/", 1)[0] for name in files if name.endswith("/takeover.mp4")}
        meta = {name.rsplit("/", 1)[0] for name in files if name.endswith("/meta.json")}
        car = {name.rsplit("/", 1)[0] for name in files if name.endswith("/carState.csv")}
        if video != meta or video != car:
            raise ValueError(f"ADAS video/meta/carState mismatch: {len(video)} / {len(meta)} / {len(car)}")
        metadata_files = ["annotations/clip_final_labels.csv", *sorted(name + "/meta.json" for name in meta)]
        telemetry_files = sorted(name + "/carState.csv" for name in car)
        video_files = sorted(name + "/takeover.mp4" for name in video)
    result = {
        "repo_id": repo_id,
        "revision": info.sha,
        "gated": info.gated,
        "units": len(video_files),
        "video_bytes": sum(files[name] or 0 for name in video_files),
        "telemetry_bytes": sum(files[name] or 0 for name in telemetry_files),
        "metadata_bytes": sum(files[name] or 0 for name in metadata_files),
        "metadata_files": metadata_files,
        "telemetry_files": telemetry_files,
        "video_files": video_files,
        "files": files,
    }
    (output / "inventory.json").write_text(json.dumps(result, indent=2) + "\n")
    for name in metadata_files if fetch_metadata else []:
        hf_hub_download(repo_id, name, repo_type="dataset", revision=info.sha, local_dir=output)
    if fetch_telemetry:
        if raw_dir is None:
            raise ValueError("--raw-dir is required when fetching telemetry")
        raw_dir.mkdir(parents=True, exist_ok=True)
        for name in telemetry_files:
            hf_hub_download(repo_id, name, repo_type="dataset", revision=info.sha, local_dir=raw_dir)
    return {key: result[key] for key in ("revision", "gated", "units", "video_bytes", "telemetry_bytes", "metadata_bytes")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=sorted(REPOS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fetch-metadata", action="store_true")
    parser.add_argument("--fetch-telemetry", action="store_true")
    parser.add_argument("--raw-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(inventory(args.source, args.output, args.fetch_metadata, args.fetch_telemetry, args.raw_dir), indent=2))


if __name__ == "__main__":
    main()
