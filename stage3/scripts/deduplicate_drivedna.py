"""Detect telemetry overlaps with BATON before final DriveDNA selection.

Thirty 10 Hz speed values, rounded to 0.1 m/s, nominate a possible overlap.
An aligned 30-second speed/acceleration/steering comparison confirms duplicate
drives. Any candidate touching BATON validation is conservatively excluded even
if the longer match diverges, because validation leakage is more costly than
losing one candidate drive. This step uses telemetry only.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from stage3.utils.config import read_jsonl


WINDOW = 30


def _values(path: Path, decimate: bool = False) -> np.ndarray:
    frame = pd.read_csv(path, usecols=["vEgo", "aEgo", "steeringAngleDeg"])
    values = frame.to_numpy(np.float32)
    return values[::10] if decimate else values


def select(baton_root: Path, baton_val_manifest: Path, raw_dir: Path,
           filter_report: Path, output: Path) -> dict:
    report = json.loads(filter_report.read_text())
    val_routes = {row["metadata"]["source_clip_id"] for row in read_jsonl(baton_val_manifest)}
    baton, fingerprints = {}, defaultdict(list)
    for path in sorted(baton_root.glob("*/vehicle_dynamics.csv")):
        route = path.parent.name
        values = _values(path, decimate=True)
        baton[route] = values
        speed = np.rint(values[:, 0] * 10).astype(np.int16)
        for i in range(0, len(speed) - WINDOW + 1, 10):
            part = speed[i:i + WINDOW]
            if part.max() - part.min() > 10:
                fingerprints[part.tobytes()].append((route, i))
    pairs = defaultdict(list)
    eligible = [row for row in report["drives"] if not row["reasons"]]
    for row in eligible:
        drive = "/".join((row["car_model"], row["driver"], row["drive"]))
        values = _values(raw_dir / row["csv_path"])
        speed = np.rint(values[:, 0] * 10).astype(np.int16)
        for i in range(len(speed) - WINDOW + 1):
            part = speed[i:i + WINDOW]
            if part.max() - part.min() <= 10:
                continue
            for route, j in fingerprints.get(part.tobytes(), ()):
                pairs[(drive, route)].append((i, j))
    details, confirmed, val_candidates = [], set(), set()
    drive_cache = {}
    for (drive, route), hits in sorted(pairs.items()):
        if drive not in drive_cache:
            name = drive.rsplit("/", 1)[-1] + ".csv"
            drive_cache[drive] = _values(raw_dir / "Dataset" / drive / name)
        dna, base = drive_cache[drive], baton[route]
        comparisons = []
        for i, j in hits[:5]:
            length = min(300, len(dna) - i, len(base) - j)
            diff = np.abs(dna[i:i + length] - base[j:j + length])
            comparisons.append((length, *(float(x) for x in diff.mean(0))))
        is_confirmed = any(n >= 100 and v < .15 and a < .2 and steer < 5
                           for n, v, a, steer in comparisons)
        if is_confirmed:
            confirmed.add(drive)
        if route in val_routes:
            val_candidates.add(drive)
        details.append({"drive": drive, "baton_route": route, "matching_speed_windows": len(hits),
                        "confirmed_30s_duplicate": is_confirmed, "touches_baton_validation": route in val_routes,
                        "comparisons": comparisons})
    excluded = confirmed | val_candidates
    selected = [row for row in eligible if "/".join((row["car_model"], row["driver"], row["drive"])) not in excluded]
    result = {
        "initial_eligible_drives": len(eligible),
        "confirmed_duplicate_drives": sorted(confirmed),
        "validation_overlap_candidates": sorted(val_candidates),
        "excluded_drive_ids": sorted(excluded),
        "accepted_drives": len(selected),
        "accepted_hours": sum(row["overlap_s"] for row in selected) / 3600,
        "accepted_video_files": [row["video_path"] for row in selected],
        "overlap_pairs": details,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "selection_report.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "selected_video_files.txt").write_text("".join(name + "\n" for name in result["accepted_video_files"]))
    return {key: result[key] for key in ("initial_eligible_drives", "confirmed_duplicate_drives", "validation_overlap_candidates", "accepted_drives", "accepted_hours")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baton-root", type=Path, required=True)
    parser.add_argument("--baton-val-manifest", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--filter-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(select(args.baton_root, args.baton_val_manifest, args.raw_dir,
                            args.filter_report, args.output), indent=2))


if __name__ == "__main__":
    main()
