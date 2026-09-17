from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import savgol_filter

from .adapters.base import Signals


def _interpolate(source_t: np.ndarray, values: np.ndarray | None, target_t: np.ndarray) -> np.ndarray:
    if values is None:
        return np.full(len(target_t), np.nan, dtype=np.float32)
    good = np.isfinite(source_t) & np.isfinite(values)
    if good.sum() < 2:
        return np.full(len(target_t), np.nan, dtype=np.float32)
    result = np.interp(target_t, source_t[good], values[good], left=np.nan, right=np.nan)
    return result.astype(np.float32)


def _smooth(values: np.ndarray, times: np.ndarray, seconds: float, polyorder: int = 2) -> np.ndarray:
    finite = np.isfinite(values)
    if finite.sum() < 3:
        return values.copy()
    filled = np.interp(times, times[finite], values[finite])
    dt = float(np.median(np.diff(times)))
    window = max(polyorder + 2, int(round(seconds / max(dt, 1e-6))))
    window += 1 - window % 2
    window = min(window, len(values) if len(values) % 2 else len(values) - 1)
    if window <= polyorder:
        return filled.astype(np.float32)
    result = savgol_filter(filled, window, min(polyorder, window - 1), mode="interp")
    result[~finite] = np.nan
    return result.astype(np.float32)


def _speed_acceleration(speed: np.ndarray, times: np.ndarray, seconds: float, polyorder: int = 2) -> tuple[np.ndarray, np.ndarray]:
    smoothed = _smooth(speed, times, seconds, polyorder)
    valid = np.isfinite(smoothed)
    result = np.full_like(smoothed, np.nan)
    if valid.sum() >= 3:
        result[valid] = np.gradient(smoothed[valid], times[valid])
    quality = valid.copy()
    if len(quality):
        quality[[0, -1]] = False
    quality &= np.isfinite(result) & (np.r_[0.0, np.diff(times)] < 0.25)
    return result.astype(np.float32), quality


def make_targets(signals: Signals, frame_times: np.ndarray, cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    """Create direct-acceleration primaries and separate speed derivatives."""
    frame_times = np.asarray(frame_times, dtype=np.float64)
    speed = _interpolate(signals.t, signals.v, frame_times)
    direct = _interpolate(signals.t, signals.a_long, frame_times)
    steering = _interpolate(signals.t, signals.steering_angle, frame_times)
    yaw = _interpolate(signals.t, signals.yaw_rate, frame_times)
    scales = cfg.get("smoothing_seconds", [0.5, 1.5])
    if len(scales) != 2:
        raise ValueError("targets.smoothing_seconds must contain two scales")
    polyorder = int(cfg.get("savgol_polyorder", 2))
    a_direct = [_smooth(direct, frame_times, float(scale), polyorder) for scale in scales]
    derived = [_speed_acceleration(speed, frame_times, float(scale), polyorder) for scale in scales]
    valid_accel = np.isfinite(a_direct[0]) & np.isfinite(a_direct[1])
    if signals.a_long is None and cfg.get("require_direct_acceleration", True):
        raise ValueError("Direct longitudinal acceleration is required for training")
    stopped_threshold = float(cfg.get("stopped_speed_mps", 0.15))
    return {
        "a_long_s1": a_direct[0],
        "a_long_s2": a_direct[1],
        "a_dvdt_s1": derived[0][0],
        "a_dvdt_s2": derived[1][0],
        "speed": speed,
        "steering_angle": steering,
        "yaw_rate_aux": yaw,
        "stopped": (speed <= stopped_threshold).astype(np.float32),
        "valid_accel": valid_accel,
        "valid_accel_speed": derived[0][1] & derived[1][1],
        "valid_speed": np.isfinite(speed),
        "valid_steer": np.isfinite(steering) & (speed > stopped_threshold),
        "valid_yaw": np.isfinite(yaw),
    }
