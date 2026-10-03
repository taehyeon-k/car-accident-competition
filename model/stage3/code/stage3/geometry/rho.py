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
    """Interval mean a/v, correcting tracked eta=v/Z for changing depth.

    With linearly varying speed, Z1=Z0-dt*(v0+v1)/2, hence
    v1/v0 = eta1/eta0 * (1-dt*eta0/2)/(1+dt*eta1/2).
    Reject intervals crossing zero speed or nonpositive depth.
    """
    rho = np.zeros_like(current, dtype=np.float32)
    if not np.isfinite(delta_t) or delta_t <= 0:
        return rho, np.zeros_like(current, dtype=bool)
    numerator = 1 - 0.5 * delta_t * previous
    denominator = 1 + 0.5 * delta_t * current
    valid = (np.isfinite(previous) & np.isfinite(current)
             & (np.abs(previous) > eps) & (np.abs(current) > eps)
             & (previous * current > 0) & (numerator > 0) & (denominator > 0))
    rho[valid] = (np.log(np.abs(current[valid])) - np.log(np.abs(previous[valid]))
                  + np.log(numerator[valid]) - np.log(denominator[valid])) / delta_t
    return rho, valid


def robust_pool(values: np.ndarray, valid: np.ndarray) -> tuple[float, float, float]:
    selected = values[valid & np.isfinite(values)]
    if len(selected) == 0:
        return 0.0, 0.0, 0.0
    center = float(np.median(selected))
    scale = float(1.4826 * np.median(np.abs(selected - center)))
    inlier = float(np.mean(np.abs(selected - center) <= max(2.5 * scale, 1e-3)))
    return center, inlier, scale
