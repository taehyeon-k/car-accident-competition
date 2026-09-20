"""Metrics. FPS is touched only by ``official_metrics``, after predictions exist."""
from __future__ import annotations

import numpy as np

TASK_WEIGHTS = (0.35, 0.35, 0.15, 0.15)


def macro_f1(target, prediction):
    scores = []
    for cls in (0, 1):
        tp = sum(t == cls and p == cls for t, p in zip(target, prediction))
        fp = sum(t != cls and p == cls for t, p in zip(target, prediction))
        fn = sum(t == cls and p != cls for t, p in zip(target, prediction))
        scores.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(scores))


def error_distribution(errors, prefix):
    """Median / mean / P25 / P50 / P75 / P90 (prompt v2 section 17)."""
    errors = np.asarray(errors, dtype=np.float64)
    out = {f"{prefix}_mean": float(errors.mean()), f"{prefix}_median": float(np.median(errors))}
    for q in (25, 50, 75, 90):
        out[f"{prefix}_p{q}"] = float(np.percentile(errors, q))
    return out


def fpsblind_metrics(predictions):
    """Selection metric. Uses frame counts only, never seconds."""
    entry_frames = np.asarray([abs(x["entry_frame"] - x["entry_gt"]) for x in predictions], float)
    collision_frames = np.asarray([abs(x["collision_frame"] - x["collision_gt"]) for x in predictions], float)
    denom = np.asarray([max(x["num_available_frames"] - 1, 1) for x in predictions], float)
    entry_norm, collision_norm = entry_frames / denom, collision_frames / denom

    result = {}
    result.update(error_distribution(entry_frames, "entry_abs_frames"))
    result.update(error_distribution(collision_frames, "collision_abs_frames"))
    result.update(error_distribution(entry_norm, "entry_normalized"))
    result.update(error_distribution(collision_norm, "collision_normalized"))
    for threshold in (0.005, 0.01, 0.02, 0.05):
        key = str(threshold * 100).rstrip("0").rstrip(".").replace(".", "p")
        result[f"entry_p_within_{key}pct"] = float((entry_norm <= threshold).mean())
        result[f"collision_p_within_{key}pct"] = float((collision_norm <= threshold).mean())
    side = macro_f1([x["entry_side_gt"] for x in predictions], [x["entry_side"] for x in predictions])
    evasion = macro_f1([x["evasion_gt"] for x in predictions], [x["evasion_space"] for x in predictions])
    we, wc, ws, wv = TASK_WEIGHTS
    result.update(f1_entry_side_macro=side, f1_evasion_space_macro=evasion)
    result["fpsblind_selection_score"] = (we * result["entry_p_within_1pct"]
                                          + wc * result["collision_p_within_1pct"]
                                          + ws * side + wv * evasion)
    return result


def official_metrics(predictions, rows):
    """Offline DACON-style +-0.3 s evaluator. Strictly post-prediction."""
    metadata = {x["sample_id"]: x for x in rows}
    entry, collision = [], []
    for pred in predictions:
        fps = float(metadata[pred["sample_id"]]["native_fps"])
        entry.append(abs(pred["entry_frame"] - pred["entry_gt"]) / fps <= 0.300001)
        collision.append(abs(pred["collision_frame"] - pred["collision_gt"]) / fps <= 0.300001)
    side = macro_f1([x["entry_side_gt"] for x in predictions], [x["entry_side"] for x in predictions])
    evasion = macro_f1([x["evasion_gt"] for x in predictions], [x["evasion_space"] for x in predictions])
    we, wc, ws, wv = TASK_WEIGHTS
    result = {"acc_entry_0.3s": float(np.mean(entry)), "acc_collision_0.3s": float(np.mean(collision)),
              "f1_entry_side_macro": side, "f1_evasion_space_macro": evasion}
    result["competition_score"] = (we * result["acc_entry_0.3s"] + wc * result["acc_collision_0.3s"]
                                   + ws * side + wv * evasion)
    return result


def by_source(predictions, rows):
    """Per-dataset normalized error, for failure analysis."""
    metadata = {x["sample_id"]: x for x in rows}
    groups = {}
    for pred in predictions:
        source = str(metadata[pred["sample_id"]].get("source_id", "unknown")).split(":", 1)[0]
        groups.setdefault(source, []).append(pred)
    out = {}
    for source, items in sorted(groups.items()):
        denom = np.asarray([max(x["num_available_frames"] - 1, 1) for x in items], float)
        out[source] = {
            "n": len(items),
            "entry_normalized_mae": float((np.asarray([abs(x["entry_frame"] - x["entry_gt"]) for x in items], float) / denom).mean()),
            "collision_normalized_mae": float((np.asarray([abs(x["collision_frame"] - x["collision_gt"]) for x in items], float) / denom).mean()),
        }
    return out
