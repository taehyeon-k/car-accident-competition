from __future__ import annotations

import numpy as np

from .rho import robust_pool


def physics_vector(
    omega: np.ndarray,
    rotation_inlier: float,
    rotation_residual: float,
    rho2: np.ndarray,
    rho2_valid: np.ndarray,
    rho4: np.ndarray,
    rho4_valid: np.ndarray,
    eta: np.ndarray,
    derotated: np.ndarray,
    confidence: np.ndarray,
    dt: float,
    pitch_proxy: float = 0.0,
) -> np.ndarray:
    r2, i2, s2 = robust_pool(rho2, rho2_valid)
    r4, i4, s4 = robust_pool(rho4, rho4_valid)
    h = eta.shape[0]
    bands = [float(np.median(eta[a:b])) for a, b in ((0, h // 3), (h // 3, 2 * h // 3), (2 * h // 3, h))]
    magnitude = np.linalg.norm(derotated, axis=0) / max(dt, 1e-6)
    static = confidence > 0.2
    static_magnitude = float(np.median(magnitude[static])) if static.any() else 0.0
    still_fraction = float(np.mean(magnitude < 0.5))
    # Monocular speed/height proxy from lower-image expansion.
    lower = eta[2 * h // 3 :]
    vh = float(np.median(np.maximum(lower, 0))) if lower.size else 0.0
    vh_quality = float(np.mean(np.isfinite(lower)))
    return np.asarray([
        *omega, rotation_inlier, rotation_residual, r2, r4, i2, i4, s2, s4,
        np.log(vh + 1e-6), vh_quality, *bands, static_magnitude, still_fraction,
        pitch_proxy, dt,
    ], dtype=np.float32)
