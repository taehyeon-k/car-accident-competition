from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import savgol_filter

from .adapters.base import Signals


def _interpolate(source_t: np.ndarray, values: np.ndarray | None, target_t: np.ndarray,
                 valid: np.ndarray | None = None, max_gap: float = 0.25) -> np.ndarray:
    result = np.full(len(target_t), np.nan, dtype=np.float32)
    if values is None:
        return result
    good = np.isfinite(values)
    if valid is not None:
        good &= np.asarray(valid, bool)
    right = np.searchsorted(source_t, target_t).clip(0, len(source_t) - 1)
    left = (right - 1).clip(0, len(source_t) - 1)
    exact = (source_t[right] == target_t) & good[right]
    result[exact] = values[right[exact]]
    span = source_t[right] - source_t[left]
    between = (~exact & good[left] & good[right] & (span > 0) & (span <= max_gap)
               & (target_t >= source_t[left]) & (target_t <= source_t[right]))
    fraction = (target_t[between] - source_t[left[between]]) / span[between]
    result[between] = values[left[between]] * (1 - fraction) + values[right[between]] * fraction
    return result


def _valid_runs(values: np.ndarray, times: np.ndarray, max_gap: float = 0.25):
    valid = np.isfinite(values) & np.isfinite(times)
    start = None
    for i in range(len(values)):
        continuous = i == 0 or 0 < times[i] - times[i - 1] <= max_gap
        if start is not None and (not valid[i] or not continuous):
            yield start, i
            start = None
        if valid[i] and start is None:
            start = i
    if start is not None:
        yield start, len(values)


def _smooth(values: np.ndarray, times: np.ndarray, seconds: float, polyorder: int = 2,
            max_gap: float = 0.25) -> np.ndarray:
    result = values.copy()
    for start, stop in _valid_runs(values, times, max_gap):
        if stop - start < 3:
            continue
        dt = float(np.median(np.diff(times[start:stop])))
        window = max(polyorder + 2, int(round(seconds / dt)))
        window += 1 - window % 2
        window = min(window, stop - start if (stop - start) % 2 else stop - start - 1)
        if window > polyorder:
            result[start:stop] = savgol_filter(values[start:stop], window, polyorder, mode="interp")
    return result.astype(np.float32)


def _speed_acceleration(speed: np.ndarray, times: np.ndarray, seconds: float, polyorder: int = 2,
                        max_gap: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    smoothed = _smooth(speed, times, seconds, polyorder, max_gap)
    result = np.full_like(smoothed, np.nan)
    quality = np.zeros(len(speed), bool)
    for start, stop in _valid_runs(smoothed, times, max_gap):
        if stop - start >= 3:
            result[start:stop] = np.gradient(smoothed[start:stop], times[start:stop])
            quality[start + 1:stop - 1] = True
    return result.astype(np.float32), quality & np.isfinite(result)


def make_targets(signals: Signals, frame_times: np.ndarray, cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    """Create direct-acceleration primaries and separate speed derivatives."""
    frame_times = np.asarray(frame_times, dtype=np.float64)
    max_gap = float(cfg.get("max_signal_gap_seconds", 0.25))
    if max_gap <= 0:
        raise ValueError("max_signal_gap_seconds must be positive")
    speed = _interpolate(signals.t, signals.v, frame_times, signals.valid, max_gap)
    direct = _interpolate(signals.t, signals.a_long, frame_times, signals.valid, max_gap)
    steering = _interpolate(signals.t, signals.steering_angle, frame_times, signals.valid, max_gap)
    yaw = _interpolate(signals.t, signals.yaw_rate, frame_times, signals.valid, max_gap)
    scales = cfg.get("smoothing_seconds", [0.5, 1.5])
    if len(scales) != 2:
        raise ValueError("targets.smoothing_seconds must contain two scales")
    polyorder = int(cfg.get("savgol_polyorder", 2))
    a_direct = [_smooth(direct, frame_times, float(scale), polyorder, max_gap) for scale in scales]
    derived = [_speed_acceleration(speed, frame_times, float(scale), polyorder, max_gap) for scale in scales]
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
