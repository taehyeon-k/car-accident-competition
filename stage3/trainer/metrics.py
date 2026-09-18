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


def classification_metrics(pred_accel: np.ndarray, true_accel: np.ndarray, pred_steer: np.ndarray, true_steer: np.ndarray,
                           valid_accel: np.ndarray | None = None, valid_steer: np.ndarray | None = None) -> dict[str, float]:
    """Score aligned frame labels; steering excludes ground-truth STOPPED frames.

    Predictions still contain a steering label for every frame. Empty eligible
    sets score zero, consistent with zero_division=0.
    """
    pred_accel, true_accel, pred_steer, true_steer = map(np.asarray, (pred_accel, true_accel, pred_steer, true_steer))
    if not (pred_accel.shape == true_accel.shape == pred_steer.shape == true_steer.shape):
        raise ValueError("All label arrays must have the same shape")
    if not np.isin(pred_steer, STEER_LABELS).all():
        raise ValueError("A valid steer_label is required for every frame, including STOPPED frames")
    amask = np.ones(true_accel.shape, bool) if valid_accel is None else np.asarray(valid_accel, bool)
    smask = np.ones(true_accel.shape, bool) if valid_steer is None else np.asarray(valid_steer, bool)
    if amask.shape != true_accel.shape or smask.shape != true_accel.shape:
        raise ValueError("Validity masks must match label shapes")
    smask = smask & (true_accel != "STOPPED")
    pred_steer, true_steer = pred_steer[smask], true_steer[smask]
    pred_accel, true_accel = pred_accel[amask], true_accel[amask]
    result = {
        "acceleration_macro_f1": float(f1_score(true_accel, pred_accel, labels=ACCEL_LABELS, average="macro", zero_division=0)) if len(true_accel) else 0.0,
        "steering_macro_f1": float(f1_score(true_steer, pred_steer, labels=STEER_LABELS, average="macro", zero_division=0)) if len(true_steer) else 0.0,
    }
    per_class = f1_score(true_accel, pred_accel, labels=ACCEL_LABELS, average=None, zero_division=0) if len(true_accel) else np.zeros(len(ACCEL_LABELS))
    result.update({f"acceleration_f1_{label.lower()}": float(value) for label, value in zip(ACCEL_LABELS, per_class)})
    matrix = confusion_matrix(true_accel, pred_accel, labels=ACCEL_LABELS) if len(true_accel) else np.zeros((4, 4))
    result["stopped_as_constant"] = float(matrix[3, 2] / max(matrix[3].sum(), 1))
    result["constant_as_stopped"] = float(matrix[2, 3] / max(matrix[2].sum(), 1))
    for seconds in (0.5, 1.0):
        value, delay = boundary_f1(pred_accel, true_accel, round(seconds * 10))
        result[f"boundary_f1_{seconds:.1f}s"] = value
        result[f"boundary_delay_{seconds:.1f}s"] = delay
    return result
