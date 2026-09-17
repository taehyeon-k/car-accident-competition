from __future__ import annotations

import numpy as np


def radial_expansion(flow: np.ndarray, foe: np.ndarray, dt: float) -> np.ndarray:
    """Per-second radial expansion coefficient around the FOE."""
    _, h, w = flow.shape
    y, x = np.mgrid[:h, :w]
    dx, dy = x - foe[0], y - foe[1]
    radius2 = dx * dx + dy * dy
    return ((flow[0] * dx + flow[1] * dy) / np.maximum(radius2, 4.0) / max(dt, 1e-6)).astype(np.float32)


def rho_from_expansion(previous: np.ndarray, current: np.ndarray, delta_t: float, eps: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a/v as temporal log-change of translation expansion."""
    valid = np.isfinite(previous) & np.isfinite(current) & (np.abs(previous) > eps) & (np.abs(current) > eps)
    rho = np.zeros_like(current, dtype=np.float32)
    same_sign = valid & (previous * current > 0)
    rho[same_sign] = (np.log(np.abs(current[same_sign])) - np.log(np.abs(previous[same_sign]))) / max(delta_t, 1e-6)
    return rho, same_sign


def robust_pool(values: np.ndarray, valid: np.ndarray) -> tuple[float, float, float]:
    selected = values[valid & np.isfinite(values)]
    if len(selected) == 0:
        return 0.0, 0.0, 0.0
    center = float(np.median(selected))
    scale = float(1.4826 * np.median(np.abs(selected - center)))
    inlier = float(np.mean(np.abs(selected - center) <= max(2.5 * scale, 1e-3)))
    return center, inlier, scale
