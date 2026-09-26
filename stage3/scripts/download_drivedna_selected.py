"""Download only DriveDNA videos selected by ``filter_drivedna``.

Hugging Face downloads are revision-pinned and resumable. The accepted list is
checked against the inventory before any transfer.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import hf_hub_download


def download(metadata_dir: Path, filtered_dir: Path, raw_dir: Path, workers: int,
             selection_report: Path | None = None) -> dict:
    inventory = json.loads((metadata_dir / "inventory.json").read_text())
    report = json.loads((filtered_dir / "filter_report.json").read_text())
    if inventory["revision"] != report["revision"]:
        raise ValueError("Inventory and filter report refer to different revisions")
    if selection_report is None:
        names = (filtered_dir / "accepted_video_files.txt").read_text().splitlines()
        expected_count = report["accepted_drives"]
    else:
        selection = json.loads(selection_report.read_text())
        names = selection["accepted_video_files"]
        expected_count = selection["accepted_drives"]
    if len(names) != expected_count or len(names) != len(set(names)):
        raise ValueError("Accepted video list does not match filter report")
    if any(name not in inventory["video_files"] for name in names):
        raise ValueError("Accepted list contains a file outside the inventory")
    raw_dir.mkdir(parents=True, exist_ok=True)

    def fetch(name: str) -> tuple[str, int]:
        path = Path(hf_hub_download(inventory["repo_id"], name, repo_type="dataset",
                                    revision=inventory["revision"], local_dir=raw_dir))
        expected = inventory["files"][name]
        if expected is not None and path.stat().st_size != expected:
            raise ValueError(f"Incomplete download: {name}")
        return name, path.stat().st_size

    done = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed({pool.submit(fetch, name): name for name in names}):
            name, size = future.result()
            done[name] = size
            print(f"{len(done)}/{len(names)} {name} {size}", flush=True)
    return {"downloaded": len(done), "bytes": sum(done.values())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--filtered-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--selection-report", type=Path,
                        help="deduplicated selection_report.json; recommended for BATON comparison")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    print(json.dumps(download(args.metadata_dir, args.filtered_dir, args.raw_dir, args.workers, args.selection_report)))


if __name__ == "__main__":
    main()
