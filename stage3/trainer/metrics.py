from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from .decoder import ACCEL_LABELS, STEER_LABELS


def boundary_times(labels: np.ndarray) -> np.ndarray:
    return np.flatnonzero(labels[1:] != labels[:-1]) + 1


def boundary_f1(predicted: np.ndarray, target: np.ndarray, tolerance_frames: int) -> tuple[float, float]:
    p, t = boundary_times(predicted), boundary_times(target)
    used: set[int] = set()
    delays = []
    for value in p:
        candidates = [(abs(int(value - other)), i, other) for i, other in enumerate(t) if i not in used and abs(value - other) <= tolerance_frames]
        if candidates:
            _, i, other = min(candidates)
            used.add(i)
            delays.append(float(value - other) / 10.0)
    precision = len(used) / max(len(p), 1)
    recall = len(used) / max(len(t), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return f1, float(np.mean(delays)) if delays else 0.0


def classification_metrics(pred_accel: np.ndarray, true_accel: np.ndarray, pred_steer: np.ndarray, true_steer: np.ndarray) -> dict[str, float]:
    result = {
        "acceleration_macro_f1": float(f1_score(true_accel, pred_accel, labels=ACCEL_LABELS, average="macro", zero_division=0)),
        "steering_macro_f1": float(f1_score(true_steer, pred_steer, labels=STEER_LABELS, average="macro", zero_division=0)),
    }
    per_class = f1_score(true_accel, pred_accel, labels=ACCEL_LABELS, average=None, zero_division=0)
    result.update({f"acceleration_f1_{label.lower()}": float(value) for label, value in zip(ACCEL_LABELS, per_class)})
    matrix = confusion_matrix(true_accel, pred_accel, labels=ACCEL_LABELS)
    result["stopped_as_constant"] = float(matrix[3, 2] / max(matrix[3].sum(), 1))
    result["constant_as_stopped"] = float(matrix[2, 3] / max(matrix[2].sum(), 1))
    for seconds in (0.5, 1.0):
        value, delay = boundary_f1(pred_accel, true_accel, round(seconds * 10))
        result[f"boundary_f1_{seconds:.1f}s"] = value
        result[f"boundary_delay_{seconds:.1f}s"] = delay
    return result
