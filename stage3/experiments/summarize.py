"""Collect runs/stage3_v2/*/metrics.json into CSV/JSON/Markdown and draw curves."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "runs/stage3_v2"
OUT = REPO / "stage3/experiments/results"
CLASSES = ["accelerating", "decelerating", "constant", "stopped"]
STEER = ["left", "right", "straight"]


def describe(overrides: list[str]) -> dict:
    o = dict(x.split("=", 1) for x in overrides if "=" in x)
    return {
        "motion": o.get("model.motion_encoder.type", "baseline"),
        "visual": o.get("model.visual.mode", "none") if o.get("model.visual.enabled") == "true" else "none",
        "fusion": o.get("model.fusion.type", "concat"),
        "temporal": o.get("model.temporal.type", "tcn"),
        "refinement": "yes" if o.get("model.refinement.enabled") == "true" else "no",
        "aux": ",".join(k.split(".")[-1] for k, v in o.items() if k.startswith("model.auxiliary.") and v == "true") or "base",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(ROOT.glob("*/metrics.json")):
        m = json.loads(path.read_text())
        v, tf, tc = m["val"], m["train_full"], m["train_at_best"]
        row = {"id": m["name"], **describe(m["overrides"]),
               "train_crop_f1": tc.get("competition_score"), "train_full_f1": tf["competition_score"],
               "val_f1": v["competition_score"], "gap_full": tf["competition_score"] - v["competition_score"],
               "val_acc_f1": v["acceleration_macro_f1"], "val_steer_f1": v["steering_macro_f1"],
               **{f"val_{c}": v[f"acceleration_f1_{c}"] for c in CLASSES},
               **{f"val_{c}": v[f"steering_f1_{c}"] for c in STEER},
               "accel_mae": v["acceleration_mae"], "speed_mae": v["speed_mae"], "steer_mae": v["steering_angle_mae"],
               "val_loss": v.get("loss"), "params": m["trainable_params"], "train_vram_mb": m["train_peak_vram_mb"],
               "eval_ms_per_frame": m["eval_ms_per_frame"], "best_epoch": m["best_epoch"], "epochs": m["epochs_run"],
               "train_minutes": m["train_seconds"] / 60}
        rows.append(row)
        history = [json.loads(x) for x in (path.parent / "history.jsonl").read_text().splitlines() if x.strip()]
        epochs = [h["epoch"] for h in history]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        axes[0].plot(epochs, [h.get("train/competition_score") for h in history], label="train (crops)")
        axes[0].plot(epochs, [h.get("val/competition_score") for h in history], label="val")
        axes[0].set_title(f"{m['name']} competition F1"); axes[0].legend()
        axes[1].plot(epochs, [h.get("train/loss") for h in history], label="train")
        axes[1].plot(epochs, [h.get("val/loss") for h in history], label="val")
        axes[1].set_title("loss"); axes[1].legend()
        fig.tight_layout(); fig.savefig(path.parent / "curves.png", dpi=90); plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
        for ax, key in zip(axes, ("confusion_acceleration", "confusion_steering")):
            matrix = np.asarray(m[key]["matrix"], float)
            norm = matrix / np.maximum(matrix.sum(1, keepdims=True), 1)
            ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
            labels = [x[:5] for x in m[key]["labels"]]
            ax.set_xticks(range(len(labels)), labels); ax.set_yticks(range(len(labels)), labels)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=8)
            ax.set_xlabel("pred"); ax.set_ylabel("true")
        fig.tight_layout(); fig.savefig(path.parent / "confusion.png", dpi=90); plt.close(fig)
    if not rows:
        return
    with (OUT / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (OUT / "summary.json").write_text(json.dumps(rows, indent=2))
    fmt = lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)
    cols = ["id", "motion", "visual", "fusion", "temporal", "refinement", "aux", "train_crop_f1", "train_full_f1",
            "val_f1", "gap_full", "val_acc_f1", "val_steer_f1", "params", "train_vram_mb"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows]
    cls = ["id"] + [f"val_{c}" for c in CLASSES + STEER] + ["accel_mae", "speed_mae", "steer_mae"]
    lines += ["", "| " + " | ".join(cls) + " |", "|" + "---|" * len(cls)]
    lines += ["| " + " | ".join(fmt(r[c]) for c in cls) + " |" for r in rows]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
