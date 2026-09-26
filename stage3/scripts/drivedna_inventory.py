"""Record DriveDNA-Sample file metadata before downloading any video.

Usage::

    python -m stage3.scripts.drivedna_inventory --output /workspace/data/stage3/drivedna/metadata

The inventory is useful even without gated-file access. Fetching index.csv is a
separate step so an inaccessible dataset never triggers a video transfer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


REPO_ID = "HenryYHW/DriveDNA-Sample"


def inventory(output: Path, fetch_index: bool = False) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    info = HfApi().repo_info(REPO_ID, repo_type="dataset", files_metadata=True)
    files = {s.rfilename: s.size for s in info.siblings or []}
    csv = {name[:-4] for name in files if name.startswith("Dataset/") and name.endswith(".csv")}
    video = {name[:-4] for name in files if name.startswith("Dataset/") and name.endswith(".mp4")}
    if csv != video:
        raise ValueError(f"DriveDNA video/CSV mismatch: CSV-only={sorted(csv-video)}, video-only={sorted(video-csv)}")
    result = {
        "repo_id": REPO_ID,
        "revision": info.sha,
        "gated": info.gated,
        "drives": len(csv),
        "video_bytes": sum(files[name + ".mp4"] or 0 for name in csv),
        "csv_bytes": sum(files[name + ".csv"] or 0 for name in csv),
        "files": files,
    }
    (output / "inventory.json").write_text(json.dumps(result, indent=2) + "\n")
    if fetch_index:
        hf_hub_download(REPO_ID, "index.csv", repo_type="dataset", revision=info.sha, local_dir=output)
    return {key: result[key] for key in ("revision", "gated", "drives", "video_bytes", "csv_bytes")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fetch-index", action="store_true")
    args = parser.parse_args()
    print(json.dumps(inventory(args.output, args.fetch_index), indent=2))


if __name__ == "__main__":
    main()
