from __future__ import annotations

import numpy as np
import torch


ACCEL_LABELS = np.asarray(["ACCELERATING", "DECELERATING", "CONSTANT", "STOPPED"])
STEER_LABELS = np.asarray(["LEFT", "STRAIGHT", "RIGHT"])


def potts_viterbi(scores: np.ndarray, penalty: float) -> np.ndarray:
    """Maximum-score Potts path; penalty zero exactly returns framewise argmax."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("scores must have shape [time, classes]")
    if len(values) == 0:
        return np.empty(0, dtype=np.int64)
    if penalty == 0:
        return values.argmax(1)
    time, classes = values.shape
    dp = np.empty_like(values)
    back = np.zeros((time, classes), np.int64)
    dp[0] = values[0]
    transitions = -float(penalty) * (1 - np.eye(classes))
    for t in range(1, time):
        candidates = dp[t - 1][:, None] + transitions
        back[t] = candidates.argmax(0)
        dp[t] = values[t] + candidates[back[t], np.arange(classes)]
    path = np.empty(time, np.int64)
    path[-1] = dp[-1].argmax()
    for t in range(time - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def acceleration_scores(acceleration: np.ndarray, stopped_probability: np.ndarray, cfg: dict) -> np.ndarray:
    decel, accel = float(cfg["decelerating_below"]), float(cfg["accelerating_above"])
    scale = float(cfg.get("emission_scale", 0.5))
    if scale <= 0 or decel >= accel:
        raise ValueError("Require positive emission_scale and ordered acceleration thresholds")
    # Linear margins intersect exactly at the configured physical thresholds.
    scores = np.stack(((acceleration - accel) / scale,
                       (decel - acceleration) / scale,
                       np.zeros_like(acceleration)), axis=1)
    scores[:, :2] = np.where(scores[:, :2] == 0, -1e-12, scores[:, :2])
    scores -= scores.max(axis=1, keepdims=True)
    p_stop = np.clip(stopped_probability, 1e-6, 1 - 1e-6)
    threshold = float(cfg.get("stopped_probability", 0.5))
    if not 0 < threshold < 1:
        raise ValueError("stopped_probability must be between zero and one")
    stop_margin = np.log(p_stop) - np.log1p(-p_stop) - np.log(threshold / (1 - threshold))
    stop_margin = np.where(p_stop >= threshold, np.maximum(stop_margin, 1e-12), stop_margin)
    return np.concatenate((scores, stop_margin[:, None]), axis=1)


def steering_scores(angle: np.ndarray, cfg: dict) -> np.ndarray:
    threshold = float(cfg["threshold_deg"])
    scale = float(cfg.get("emission_scale_deg", max(threshold, 1.0)))
    if threshold < 0 or scale <= 0:
        raise ValueError("Require nonnegative steering threshold and positive emission scale")
    scores = np.stack(((angle - threshold) / scale, np.zeros_like(angle),
                       (-angle - threshold) / scale), axis=1)
    scores[:, [0, 2]] = np.where(scores[:, [0, 2]] == 0, -1e-12, scores[:, [0, 2]])
    return scores


def decode_predictions(outputs: dict[str, torch.Tensor] | dict[str, np.ndarray], cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    def array(name: str) -> np.ndarray:
        value = outputs[name]
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy()
        value = np.asarray(value)
        return value[0] if value.ndim > (2 if name == "acceleration" else 1) else value

    accel_two = array("acceleration")
    acceleration = accel_two.mean(-1)
    stop = 1.0 / (1.0 + np.exp(-array("stop_logit")))
    steering = array("steering_angle")
    accel_path = potts_viterbi(acceleration_scores(acceleration, stop, cfg["acceleration"]), cfg["acceleration"]["potts_penalty"])
    steer_path = potts_viterbi(steering_scores(steering, cfg["steering"]), cfg["steering"]["potts_penalty"])
    return ACCEL_LABELS[accel_path], STEER_LABELS[steer_path]
