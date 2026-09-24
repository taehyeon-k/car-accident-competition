"""Frame-count/source diagnostics and plots for B0 and P1–P4."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from stage2.spotting_experiments.data import read_rows
from stage2.spotting_experiments.train import official_metrics

ROOT = Path("stage2/temporal_pyramid_experiments/results")
B0 = Path("stage2/spotting_experiments/results/E3-ASFormer-349/fixed/seed0")
BINS = [(0, 150, "<150"), (150, 300, "150–300"), (300, 500, "300–500"),
        (500, 750, "500–750"), (750, 1000, "750–1000"), (1000, 10**9, ">1000")]


def source(p): return p["source_id"].split(":", 1)[0]


def stats(predictions, metadata):
    if not predictions: return {"n": 0}
    m = official_metrics(predictions, metadata)
    n = len(predictions)
    m.update(n=n, entry_normalized_mae=float(np.mean([abs(x["entry_frame"] - x["entry_gt"]) /
                    max(x["num_available_frames"] - 1, 1) for x in predictions])),
             collision_normalized_mae=float(np.mean([abs(x["collision_frame"] - x["collision_gt"]) /
                    max(x["num_available_frames"] - 1, 1) for x in predictions])))
    return m


def correlations(predictions, metadata):
    if len(predictions) < 5: return {"n": len(predictions)}
    lookup = {r["sample_id"]: r for r in metadata}
    frames = np.asarray([x["num_available_frames"] for x in predictions])
    ee = np.asarray([abs(x["entry_frame"] - x["entry_gt"]) / max(x["num_available_frames"] - 1, 1) for x in predictions])
    ce = np.asarray([abs(x["collision_frame"] - x["collision_gt"]) / max(x["num_available_frames"] - 1, 1) for x in predictions])
    eh = np.asarray([abs(x["entry_frame"] - x["entry_gt"]) / lookup[x["sample_id"]]["native_fps"] <= .300001 for x in predictions])
    ch = np.asarray([abs(x["collision_frame"] - x["collision_gt"]) / lookup[x["sample_id"]]["native_fps"] <= .300001 for x in predictions])
    # Per-video correctness is a proxy; macro-F1 is computed only for groups.
    score = .35 * eh + .35 * ch + .15 * np.asarray([x["entry_side"] == x["entry_side_gt"] for x in predictions]) + .15 * np.asarray([x["evasion_space"] == x["evasion_gt"] for x in predictions])
    def rho(v):
        r = spearmanr(frames, v).statistic
        return float(r) if np.isfinite(r) else None
    return {"n": len(predictions), "frame_count_vs_entry_normalized_error_rho": rho(ee),
            "frame_count_vs_collision_normalized_error_rho": rho(ce),
            "frame_count_vs_per_video_correctness_proxy_rho": rho(score)}


def main():
    all_rows = read_rows("/workspace/data/stage2/manifests/all.jsonl")
    results = {}
    runs = ["B0"] + [p.name for p in sorted(ROOT.iterdir()) if p.is_dir() and (p / "seed0" / "predictions.json").exists()]
    for run in runs:
        path = B0 if run == "B0" else ROOT / run / "seed0"
        if not (path / "predictions.json").exists(): continue
        predictions = json.loads((path / "predictions.json").read_text())
        metrics = json.loads((path / "metrics.json").read_text())
        results[run] = {"overall": stats(predictions, all_rows),
                        "bins": {label: stats([x for x in predictions if lo <= x["num_available_frames"] < hi], all_rows)
                                 for lo, hi, label in BINS},
                        "sources": {s: stats([x for x in predictions if source(x) == s], all_rows)
                                    for s in sorted({source(x) for x in predictions})},
                        "correlations": {"global": correlations(predictions, all_rows),
                                         **{s: correlations([x for x in predictions if source(x) == s], all_rows)
                                            for s in sorted({source(x) for x in predictions})}},
                        "runtime": metrics["runtime"],
                        "history": json.loads((path / "history.json").read_text()) if (path / "history.json").exists() else []}
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "analysis.json").write_text(json.dumps(results, indent=2) + "\n")
    if not results: return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for run, r in results.items():
        y = [r["bins"][label].get("competition_score", np.nan) for _, _, label in BINS]
        axes[0].plot(range(len(BINS)), y, marker="o", label=run)
        h = r["history"]
        if not h: continue
        if run == "B0":
            val = [x["fpsblind_selection_score"] for x in h]
        else:
            val = [x["val_fpsblind_score"] for x in h]
            axes[1].plot(range(1, len(val) + 1), [x["train_fpsblind_score"] for x in h],
                         linestyle="--", alpha=.5, label=f"{run} train")
        axes[1].plot(range(1, len(val) + 1), val, label=f"{run} val")
    axes[0].set_xticks(range(len(BINS)), [x[2] for x in BINS], rotation=30)
    axes[0].set_ylabel("Offline competition-style score")
    axes[0].set_xlabel("Original video frame count")
    axes[1].set_ylabel("FPS-blind validation selection score")
    axes[1].set_xlabel("Epoch")
    for a in axes: a.grid(alpha=.3); a.legend()
    fig.tight_layout(); fig.savefig(ROOT / "framecount_and_validation.png", dpi=140); plt.close(fig)
    print(json.dumps({k: {"overall": v["overall"], "bins": v["bins"], "sources": v["sources"]} for k, v in results.items()}, indent=2))


if __name__ == "__main__": main()
