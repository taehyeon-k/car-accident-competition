from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from .decoder import ACCEL_LABELS, STEER_LABELS, decode_predictions


def competition_score(acceleration_macro_f1: float, steering_macro_f1: float) -> float:
    """Official Stage 3 weighted score (range 0..1)."""
    return 0.7 * acceleration_macro_f1 + 0.3 * steering_macro_f1


def classification_targets(targets: dict, decoder: dict):
    """Shared train/validation labels and validity masks on one unpadded clip."""
    time = targets["time_valid"].astype(bool)
    stopped = targets["stopped"] > 0.5
    direct = 0.5 * (targets["a_long_s1"] + targets["a_long_s2"])
    angle, speed = targets["steering_angle"], targets["speed"]
    speed_valid = time & targets["valid_speed"].astype(bool) & np.isfinite(speed)
    direct_valid = time & targets["valid_accel"].astype(bool) & np.isfinite(direct)
    accel_valid = speed_valid & (stopped | direct_valid)
    steer_valid = speed_valid & ~stopped & targets["valid_steer"].astype(bool) & np.isfinite(angle)
    cfg = decoder["acceleration"]
    accel = np.where(stopped, "STOPPED", np.where(direct > cfg["accelerating_above"], "ACCELERATING",
                     np.where(direct < cfg["decelerating_below"], "DECELERATING", "CONSTANT")))
    threshold = decoder["steering"]["threshold_deg"]
    steer = np.where(angle > threshold, "LEFT", np.where(angle < -threshold, "RIGHT", "STRAIGHT"))
    return accel, steer, accel_valid, steer_valid


class TrainingCompetitionMetrics:
    """Aggregate frame counts from existing training forwards, not batch F1s.

    Training scores describe the sampled/augmented crops and the evolving live
    model. Validation continues to evaluate full clips with the EMA model.
    """

    def __init__(self):
        self.acceleration = np.zeros((len(ACCEL_LABELS), len(ACCEL_LABELS)), np.int64)
        self.steering = np.zeros((len(STEER_LABELS), len(STEER_LABELS)), np.int64)

    def update(self, outputs: dict, batch: dict, decoder: dict, accelerator=None):
        names = ("time_valid", "stopped", "a_long_s1", "a_long_s2", "steering_angle", "speed",
                 "valid_speed", "valid_accel", "valid_steer")
        targets = {name: batch[name].detach().cpu().numpy() for name in names}
        predicted = {name: outputs[name].detach().float().cpu().numpy()
                     for name in ("acceleration", "stop_logit", "steering_angle")}
        counts = []
        for i, length in enumerate(batch["lengths"].detach().cpu().tolist()):
            sample = {name: value[i, :length] for name, value in targets.items()}
            ta, ts, amask, smask = classification_targets(sample, decoder)
            pa, ps = decode_predictions({name: value[i, :length] for name, value in predicted.items()}, decoder)
            amatrix = confusion_matrix(ta[amask], pa[amask], labels=ACCEL_LABELS) if amask.any() else np.zeros_like(self.acceleration)
            smatrix = confusion_matrix(ts[smask], ps[smask], labels=STEER_LABELS) if smask.any() else np.zeros_like(self.steering)
            counts.append((amatrix, smatrix))
        if getattr(accelerator, "num_processes", 1) > 1:
            # Gather per-clip counts so Accelerate can trim padded tail samples.
            counts = accelerator.gather_for_metrics(counts, use_gather_object=True)
        for amatrix, smatrix in counts:
            self.acceleration += amatrix
            self.steering += smatrix

    def compute(self) -> dict[str, float]:
        def macro_f1(matrix):
            denominator = matrix.sum(0) + matrix.sum(1)
            f1 = np.divide(2.0 * matrix.diagonal(), denominator,
                           out=np.zeros(len(matrix)), where=denominator > 0)
            return float(f1.mean())

        accel, steer = macro_f1(self.acceleration), macro_f1(self.steering)
        return {"acceleration_macro_f1": accel, "steering_macro_f1": steer,
                "competition_score": competition_score(accel, steer)}


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
    result["competition_score"] = competition_score(result["acceleration_macro_f1"], result["steering_macro_f1"])
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
