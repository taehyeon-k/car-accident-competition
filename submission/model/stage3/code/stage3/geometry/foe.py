from __future__ import annotations

import numpy as np


def estimate_foe(flow: np.ndarray, confidence: np.ndarray, iterations: int = 3, fit_stride: int = 4) -> tuple[np.ndarray, float, float]:
    """Robust line-intersection focus of expansion in pixel coordinates."""
    _, h, w = flow.shape
    y, x = np.mgrid[:h, :w].astype(np.float64)
    u, v = flow.astype(np.float64)
    x, y, u, v, confidence = (
        value[::fit_stride, ::fit_stride] for value in (x, y, u, v, confidence)
    )
    magnitude = np.hypot(u, v)
    valid = np.isfinite(magnitude) & (magnitude > 0.05) & (confidence > 0.05)
    a = np.stack((v[valid], -u[valid]), axis=1)
    b = (v * x - u * y)[valid]
    base = confidence[valid] * np.minimum(magnitude[valid], np.percentile(magnitude[valid], 90) if valid.any() else 1)
    if len(b) < 20:
        return np.array([(w - 1) / 2, (h - 1) / 2], np.float32), 0.0, float("inf")
    weights = base
    foe = np.array([(w - 1) / 2, (h - 1) / 2])
    for _ in range(iterations):
        root = np.sqrt(weights)[:, None]
        foe = np.linalg.lstsq(a * root, b * root[:, 0], rcond=None)[0]
        residual = b - a @ foe
        scale = np.median(np.abs(residual)) * 1.4826 + 1e-6
        weights = base / np.maximum(1.0, np.abs(residual) / (2.5 * scale))
    inlier = float(np.mean(np.abs(residual) < 2.5 * scale))
    return foe.astype(np.float32), inlier, float(scale)
