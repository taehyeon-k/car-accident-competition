"""Failure analysis for one run (prompt v2 section 21).

Buckets validation errors by direction and magnitude and cross-tabulates them
against clip properties that are available without any FPS: dataset source,
clip length in frames, and where the event sits inside the clip.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .data import read_rows


def direction(pred, gt, tolerance):
    delta = pred - gt
    if abs(delta) <= tolerance:
        return "hit"
    return "early" if delta < 0 else "late"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="a results/<run> directory")
    parser.add_argument("--output")
    args = parser.parse_args()

    run = Path(args.run)
    predictions = json.loads((run / "predictions.json").read_text())
    cfg = json.loads((run / "config.json").read_text())["config"]
    rows = {r["sample_id"]: r for r in read_rows(cfg["all_manifest"])}

    records = []
    for pred in predictions:
        row = rows[pred["sample_id"]]
        span = max(pred["num_available_frames"] - 1, 1)
        # Tolerance in frames that matches the offline +-0.3 s metric for this clip.
        tolerance = 0.3 * float(row["native_fps"])
        records.append({
            "sample_id": pred["sample_id"],
            "source": str(row.get("source_id", "unknown")).split(":", 1)[0],
            "num_frames": pred["num_available_frames"],
            "entry_direction": direction(pred["entry_frame"], pred["entry_gt"], tolerance),
            "collision_direction": direction(pred["collision_frame"], pred["collision_gt"], tolerance),
            "entry_error_frames": pred["entry_frame"] - pred["entry_gt"],
            "collision_error_frames": pred["collision_frame"] - pred["collision_gt"],
            "entry_normalized_error": abs(pred["entry_frame"] - pred["entry_gt"]) / span,
            "collision_normalized_error": abs(pred["collision_frame"] - pred["collision_gt"]) / span,
            "side_correct": pred["entry_side"] == pred["entry_side_gt"],
            "evasion_correct": pred["evasion_space"] == pred["evasion_gt"],
            "gt_entry_position": pred["entry_gt"] / span,
            "gt_event_gap_frames": pred["collision_gt"] - pred["entry_gt"],
        })

    def counts(key):
        return dict(Counter(r[key] for r in records))

    def worst(key, n=8):
        ranked = sorted(records, key=lambda r: -r[key])[:n]
        return [{"sample_id": r["sample_id"], "source": r["source"], key: round(r[key], 4),
                 "num_frames": r["num_frames"], "gt_entry_position": round(r["gt_entry_position"], 3),
                 "gt_event_gap_frames": r["gt_event_gap_frames"]} for r in ranked]

    def split_by(predicate, key):
        yes = [r[key] for r in records if predicate(r)]
        no = [r[key] for r in records if not predicate(r)]
        return {"group_n": len(yes), "group_mean": float(np.mean(yes)) if yes else None,
                "rest_n": len(no), "rest_mean": float(np.mean(no)) if no else None}

    report = {
        "run": str(run.resolve()),
        "config_id": cfg.get("id"),
        "entry_direction_counts": counts("entry_direction"),
        "collision_direction_counts": counts("collision_direction"),
        "side_errors": [r["sample_id"] for r in records if not r["side_correct"]],
        "evasion_errors": [r["sample_id"] for r in records if not r["evasion_correct"]],
        "worst_entry": worst("entry_normalized_error"),
        "worst_collision": worst("collision_normalized_error"),
        # Does error track clip length, event position, or event separation?
        "entry_error_by_short_clip": split_by(lambda r: r["num_frames"] < 200, "entry_normalized_error"),
        "entry_error_by_early_event": split_by(lambda r: r["gt_entry_position"] < 0.4, "entry_normalized_error"),
        "entry_error_by_tight_gap": split_by(lambda r: r["gt_event_gap_frames"] <= 10, "entry_normalized_error"),
        "collision_error_by_tight_gap": split_by(lambda r: r["gt_event_gap_frames"] <= 10, "collision_normalized_error"),
        "per_source": {
            source: {
                "n": sum(1 for r in records if r["source"] == source),
                "entry_normalized_mae": float(np.mean([r["entry_normalized_error"] for r in records if r["source"] == source])),
                "collision_normalized_mae": float(np.mean([r["collision_normalized_error"] for r in records if r["source"] == source])),
                "side_accuracy": float(np.mean([r["side_correct"] for r in records if r["source"] == source])),
                "evasion_accuracy": float(np.mean([r["evasion_correct"] for r in records if r["source"] == source])),
            }
            for source in sorted({r["source"] for r in records})
        },
    }
    text = json.dumps(report, indent=2) + "\n"
    (Path(args.output) if args.output else run / "failure_analysis.json").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
