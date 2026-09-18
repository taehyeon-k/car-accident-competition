from __future__ import annotations

import numpy as np


def speed_over_height_proxy(expansion: np.ndarray, confidence: np.ndarray) -> tuple[float, float]:
    """Robust lower-image translation-expansion proxy for v/h and its support."""
    lower = expansion[2 * expansion.shape[0] // 3 :]
    mask = confidence[2 * expansion.shape[0] // 3 :] > 0.2
    values = lower[mask & np.isfinite(lower)]
    if not len(values):
        return 0.0, 0.0
    return float(np.median(np.maximum(values, 0))), float(len(values) / mask.size)
