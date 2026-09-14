"""Confirm every labelled sample's ENTRY->COLLISION span fits the memory cap.

The 512-frame training cap is only safe because it never has to move or drop a
label. Run this after changing the manifests or the cap.
"""

import argparse
import json

from stage2.data.cache_geometry import frame_paths
from stage2.data.joint_sampling import DEFAULT_MAX_TRAIN_FRAMES
from stage2.utils.utils import read_manifest


def check(manifest: str, max_frames: int) -> dict:
    worst = {"span": 0, "sample_id": None}
    violations = []
    rows = read_manifest(manifest)
    for row in rows:
        _, frame_ids = frame_paths(row["frames_dir"])
        entry = frame_ids.index(int(row["entry_frame"]))
        collision = frame_ids.index(int(row["collision_frame"]))
        span = collision - entry + 1
        if span > worst["span"]:
            worst = {"span": span, "sample_id": row["sample_id"]}
        if span > max_frames:
            violations.append({"sample_id": row["sample_id"], "span": span})
    return {
        "manifest": manifest,
        "samples": len(rows),
        "max_frames": max_frames,
        "longest_span": worst,
        "violations": violations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, nargs="+")
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_TRAIN_FRAMES)
    args = parser.parse_args()
    failed = False
    for manifest in args.manifest:
        report = check(manifest, args.max_frames)
        print(json.dumps(report))
        failed |= bool(report["violations"])
    if failed:
        raise SystemExit("Some samples cannot be capped without corrupting labels")


if __name__ == "__main__":
    main()
