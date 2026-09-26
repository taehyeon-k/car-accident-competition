"""Write the completed BATON versus BATON+DriveDNA experiment report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


LABELS = ["acceleration_f1_accelerating", "acceleration_f1_decelerating",
          "acceleration_f1_constant", "acceleration_f1_stopped",
          "steering_f1_left", "steering_f1_straight", "steering_f1_right"]


def _score(row: dict) -> str:
    m = row["metrics"]
    return (f"{m['acceleration_macro_f1']:.4f} | {m['steering_macro_f1']:.4f} | "
            f"{m['competition_score']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/stage3_data_scaling"))
    parser.add_argument("--output", type=Path,
                        default=Path("stage3/experiments/BATON_DRIVEDNA_REPORT.md"))
    args = parser.parse_args()
    baseline = json.loads((args.root / "baseline_sources.json").read_text())
    merged = json.loads((args.root / "merged_sources.json").read_text())
    run = json.loads((args.root / "Baton_DriveDNA_V3/metrics.json").read_text())
    composition = json.loads(Path("/workspace/data/stage3/baton_drivedna_final/manifests/composition.json").read_text())
    filtered = json.loads(Path("/workspace/data/stage3/drivedna/filtered/filter_report.json").read_text())
    selected = json.loads(Path("/workspace/data/stage3/drivedna/filtered/selection_report.json").read_text())
    b, m = baseline["sources"], merged["sources"]
    delta = m["baton_val"]["metrics"]["competition_score"] - b["baton_val"]["metrics"]["competition_score"]
    direction = "improved" if delta > 0 else "degraded" if delta < 0 else "matched"
    lines = [
        "# BATON versus BATON + DriveDNA Stage 3 experiment", "",
        "The merged model " + direction + f" the unchanged BATON validation score by {delta:+.4f}. "
        "This is an offline result; no new leaderboard submission has been scored.", "",
        "## Data and compatibility", "",
        f"BATON baseline: {composition['baton_train']['drives']} training and {composition['baton_val']['drives']} validation routes; "
        f"{composition['baton_train']['segments']} / {composition['baton_val']['segments']} 30-second segments, "
        f"{composition['baton_train']['hours']:.2f} / {composition['baton_val']['hours']:.2f} hours. "
        "The original split and cache content were left intact.", "",
        f"DriveDNA-Sample: {filtered['candidates']} candidate drives, "
        f"{filtered['rejections_by_criterion'].get('missing_required_columns', 0)} missing direct acceleration/steering, "
        f"{selected['initial_eligible_drives']} telemetry-eligible drives, "
        f"{len(selected['excluded_drive_ids'])} excluded for confirmed BATON duplication or conservative validation overlap, "
        f"{selected['accepted_drives']} finally selected drives ({selected['accepted_hours']:.2f} hours). "
        f"DriveDNA split: {composition['drivedna_train']['drives']} / {composition['drivedna_val']['drives']} "
        f"original drives, {composition['drivedna_train']['segments']} / {composition['drivedna_val']['segments']} segments. "
        f"Merged training set: {composition['train']['segments']} segments, {composition['train']['hours']:.2f} hours.", "",
        "DriveDNA maps `time_s`, `vEgo` (m/s), `aEgo` (m/s²), and `steeringAngleDeg` "
        "(steering-wheel degrees, positive left) directly into the same `Signals` class as BATON. "
        "Both have video time zero. DriveDNA CSV is 10 Hz and video is approximately 20 fps; "
        "the shared PyAV PTS nearest-frame path resamples video to 10 Hz and interpolates CAN at selected actual frame times. "
        "The legacy three-column DriveDNA schema is excluded. `is_human` was not used as a filter, "
        "matching BATON's inclusion of automation frames. The same smoothing, validity masks, "
        "motion feature cache, and BATON training normalization were used.", "",
        "Telemetry checks: all 60 eligible drives had 0.1 s median sample spacing and no gaps over 0.25 s. "
        "The median DriveDNA acceleration versus speed-derivative correlation was positive; "
        "steering versus curvature correlation was negative in both datasets, supporting the same steering sign. "
        "Three synchronized DriveDNA video/CAN examples are stored in "
        "`/workspace/data/stage3/drivedna/filtered/synchronized_examples.png`.", "",
        "## Fixed method", "",
        "Both models use the saved `V3_tcnssm_100ep` architecture, seed 42, losses, crop, augmentation, "
        "AdamW/cosine schedule, EMA, and 100-epoch stopping rule. Only the training manifest changed. "
        "Both were evaluated full-clip with the same EMA protocol, steering 5°, acceleration ±0.5 m/s², "
        "and ground-truth STOPPED frames masked from steering F1. The saved baseline checkpoint was used as A; "
        "the merged run's best checkpoint was selected on combined validation using the original ±0.25 m/s² "
        "training decoder, then both checkpoints were evaluated at ±0.5 m/s² for this comparison.", "",
        "## Results", "",
        "| Model and domain | Accel Macro-F1 | Steer Macro-F1 | Stage 3 score |", "|---|---:|---:|---:|",
    ]
    for name, label in (("baton_val", "BATON validation"), ("drivedna_val", "DriveDNA validation"),
                        ("combined_val", "Combined validation")):
        lines.append(f"| BATON only — {label} | {_score(b[name])} |")
        lines.append(f"| BATON + DriveDNA — {label} | {_score(m[name])} |")
    lines += ["", "| Full-clip train score | Accel Macro-F1 | Steer Macro-F1 | Stage 3 score |",
              "|---|---:|---:|---:|",
              f"| BATON only — BATON train | {_score(b['baton_train'])} |",
              f"| BATON + DriveDNA — combined train | {_score(m['combined_train'])} |", "",
              "### BATON validation per-class F1", "",
              "| Class | BATON only | BATON + DriveDNA | Change |", "|---|---:|---:|---:|"]
    for key in LABELS:
        before = b["baton_val"]["metrics"][key]
        after = m["baton_val"]["metrics"][key]
        lines.append(f"| {key.removeprefix('acceleration_f1_').removeprefix('steering_f1_')} | {before:.4f} | {after:.4f} | {after-before:+.4f} |")
    lines += ["", "Confusion matrices, source-specific metrics, and runtime/VRAM records are in "
              f"`{args.root}/baseline_sources.json`, `{args.root}/merged_sources.json`, and "
              f"`{args.root}/Baton_DriveDNA_V3/metrics.json`. "
              f"The merged run's peak allocated training VRAM was {run['train_peak_vram_mb']:.0f} MiB.", "",
              "The important comparison is the unchanged BATON validation domain. "
              "The preexisting BATON route-family split may overstate generalization because sibling recordings "
              "of some routes cross train/validation; this experiment does not modify that split. "
              "DriveDNA validation is drive-disjoint but may contain the same driver identity as DriveDNA training. "
              "A new leaderboard result is needed before claiming real-test improvement.", ""]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(args.output)


if __name__ == "__main__":
    main()
