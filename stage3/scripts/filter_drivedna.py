"""Filter DriveDNA-Sample from index and CSV telemetry, before video download.

The BATON baseline only required paired video and telemetry plus the canonical
speed, direct acceleration, and steering fields. This script applies those same
contract checks, records signal diagnostics, and never downloads video.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED = ("time_s", "vEgo", "aEgo", "steeringAngleDeg")
OPTIONAL = ("yawRate", "is_human")


def _summary(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"count": 0}
    q = np.quantile(finite, [0, .001, .01, .1, .5, .9, .99, .999, 1])
    return {"count": int(len(finite)), "quantiles": dict(zip(
        ("min", "p001", "p01", "p10", "median", "p90", "p99", "p999", "max"),
        map(float, q), strict=True))}


def filter_drives(metadata_dir: Path, raw_dir: Path, output: Path) -> dict:
    inventory = json.loads((metadata_dir / "inventory.json").read_text())
    if inventory["repo_id"] != "HenryYHW/DriveDNA-Sample":
        raise ValueError("Wrong dataset inventory")
    index = pd.read_csv(metadata_dir / "index.csv")
    rows = []
    rejected = Counter()
    global_values = {key: [] for key in ("vEgo", "aEgo", "steeringAngleDeg")}
    for item in index.itertuples(index=False):
        stem = f"Dataset/{item.car_model}/{item.driver}/{item.drive}/{item.drive}"
        csv_rel, video_rel = stem + ".csv", stem + ".mp4"
        reasons = []
        if csv_rel not in inventory["files"] or video_rel not in inventory["files"]:
            reasons.append("missing_video_csv_pair")
        csv_path = raw_dir / csv_rel
        if not csv_path.is_file():
            raise FileNotFoundError(f"Required telemetry missing: {csv_path}")
        header = list(pd.read_csv(csv_path, nrows=0).columns)
        missing = sorted(set(REQUIRED) - set(header))
        if missing:
            reasons.append("missing_required_columns")
            data = None
        else:
            data = pd.read_csv(csv_path, usecols=[*REQUIRED, *(key for key in OPTIONAL if key in header)])
            for key in REQUIRED:
                data[key] = pd.to_numeric(data[key], errors="coerce")
            data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["time_s"])
            data = data.drop_duplicates("time_s", keep="first").sort_values("time_s")
            if len(data) < 2 or (np.diff(data.time_s.to_numpy()) <= 0).any():
                reasons.append("invalid_timestamps")
            for key in REQUIRED[1:]:
                if not np.isfinite(data[key].to_numpy(np.float64)).any():
                    reasons.append(f"no_finite_{key}")
        duration = float(min(item.video_s, item.csv_s))
        if not np.isfinite(duration) or duration < 2:
            reasons.append("insufficient_overlap")
        detail = {
            "car_model": item.car_model, "driver": item.driver, "drive": item.drive,
            "csv_path": csv_rel, "video_path": video_rel,
            "csv_schema": item.csv_schema, "csv_hz": int(item.csv_hz),
            "video_s": float(item.video_s), "csv_s": float(item.csv_s),
            "overlap_s": duration, "reasons": reasons,
        }
        if data is not None and len(data) >= 2:
            t = data.time_s.to_numpy(np.float64)
            detail["rows_after_time_cleaning"] = len(data)
            detail["median_dt_s"] = float(np.median(np.diff(t)))
            detail["max_dt_s"] = float(np.max(np.diff(t)))
            detail["gaps_over_0p25s"] = int((np.diff(t) > .25).sum())
            detail["signals"] = {key: _summary(data[key].to_numpy(np.float64)) for key in REQUIRED[1:]}
            if "is_human" in data:
                human = pd.to_numeric(data.is_human, errors="coerce").to_numpy(np.float64)
                detail["human_fraction"] = float(np.nanmean(human == 1))
            if "yawRate" in data:
                steer = data.steeringAngleDeg.to_numpy(np.float64)
                yaw = pd.to_numeric(data.yawRate, errors="coerce").to_numpy(np.float64)
                speed = data.vEgo.to_numpy(np.float64)
                moving = np.isfinite(steer) & np.isfinite(yaw) & (speed > 5) & (np.abs(steer) > 5)
                if moving.sum() > 10 and np.std(steer[moving]) > 0 and np.std(yaw[moving]) > 0:
                    detail["steering_yaw_correlation_moving"] = float(np.corrcoef(steer[moving], yaw[moving])[0, 1])
            speed = data.vEgo.to_numpy(np.float64)
            accel = data.aEgo.to_numpy(np.float64)
            dt = np.diff(t)
            compatible = (dt > 0) & (dt <= .25) & np.isfinite(speed[:-1]) & np.isfinite(speed[1:]) & np.isfinite(accel[:-1])
            if compatible.sum() > 10:
                dvdt = np.diff(speed)[compatible] / dt[compatible]
                a = accel[:-1][compatible]
                if np.std(dvdt) > 0 and np.std(a) > 0:
                    detail["accel_speed_derivative_correlation"] = float(np.corrcoef(a, dvdt)[0, 1])
            if not reasons:
                for key in global_values:
                    global_values[key].append(data[key].to_numpy(np.float32))
        rows.append(detail)
        for reason in reasons:
            rejected[reason] += 1
    accepted = [row for row in rows if not row["reasons"]]
    report = {
        "repo_id": inventory["repo_id"], "revision": inventory["revision"],
        "candidates": len(rows), "rejections_by_criterion": dict(rejected),
        "accepted_drives": len(accepted),
        "accepted_hours": sum(row["overlap_s"] for row in accepted) / 3600,
        "accepted_video_bytes": sum(inventory["files"][row["video_path"]] or 0 for row in accepted),
        "signal_distributions": {key: _summary(np.concatenate(values)) if values else {"count": 0}
                                 for key, values in global_values.items()},
        "drives": rows,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "filter_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "accepted_video_files.txt").write_text("".join(row["video_path"] + "\n" for row in accepted))
    return {key: report[key] for key in ("candidates", "rejections_by_criterion", "accepted_drives", "accepted_hours", "accepted_video_bytes", "signal_distributions")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(filter_drives(args.metadata_dir, args.raw_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
