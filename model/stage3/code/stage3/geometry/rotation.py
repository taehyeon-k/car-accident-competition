from __future__ import annotations

import numpy as np


def rotational_design(height: int, width: int, focal: float) -> np.ndarray:
    """Dense small-angle 3-DoF camera rotation design matrix."""
    y, x = np.mgrid[:height, :width].astype(np.float64)
    x -= (width - 1) / 2.0
    y -= (height - 1) / 2.0
    u = np.stack((x * y / focal, -(focal + x * x / focal), y), axis=-1)
    v = np.stack((focal + y * y / focal, -x * y / focal, -x), axis=-1)
    return np.stack((u, v), axis=-2)


def estimate_rotation(flow: np.ndarray, confidence: np.ndarray, focal: float, iterations: int = 3, fit_stride: int = 4) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Robustly fit angular displacement and return its optical flow."""
    _, h, w = flow.shape
    full_design = rotational_design(h, w, focal)
    design = full_design[::fit_stride, ::fit_stride].reshape(-1, 3)
    target = flow[:, ::fit_stride, ::fit_stride].transpose(1, 2, 0).reshape(-1)
    base = np.repeat(confidence[::fit_stride, ::fit_stride].reshape(-1), 2)
    finite = np.isfinite(target) & np.isfinite(base) & (base > 0.05)
    a, b, weights = design[finite], target[finite], base[finite]
    if len(b) < 20:
        omega = np.zeros(3)
    else:
        omega = np.zeros(3)
        for _ in range(iterations):
            root = np.sqrt(weights)[:, None]
            omega = np.linalg.lstsq(a * root, b * root[:, 0], rcond=None)[0]
            residual = b - a @ omega
            scale = np.median(np.abs(residual)) * 1.4826 + 1e-6
            weights = base[finite] / np.maximum(1.0, np.abs(residual) / (2.5 * scale))
    predicted = np.einsum("hwkc,c->hwk", full_design, omega).transpose(2, 0, 1)
    residual_map = np.linalg.norm(flow - predicted, axis=0)
    scale = float(np.median(residual_map[np.isfinite(residual_map)]))
    inlier = float(np.mean(residual_map < max(2.5 * scale, 0.25)))
    return omega.astype(np.float32), predicted.astype(np.float32), inlier, scale
