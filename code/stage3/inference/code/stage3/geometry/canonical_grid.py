from __future__ import annotations

import numpy as np


def resize_channel(channel: np.ndarray, output_hw: tuple[int, int], nearest: bool = False) -> np.ndarray:
    import cv2

    h, w = output_hw
    mode = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    return cv2.resize(channel.astype(np.float32), (w, h), interpolation=mode)


def canonical_tensor(
    derotated: np.ndarray,
    dt: float,
    eta: np.ndarray,
    rho: np.ndarray,
    rho_valid: np.ndarray,
    confidence: np.ndarray,
    static_valid: np.ndarray,
    foe: np.ndarray,
    output_hw: tuple[int, int] = (96, 168),
) -> np.ndarray:
    _, h, w = derotated.shape
    y, x = np.mgrid[:h, :w]
    dx = (x - foe[0]) / max(w, 1)
    dy = (y - foe[1]) / max(h, 1)
    velocity = derotated / max(dt, 1e-6)
    magnitude = np.log1p(np.linalg.norm(velocity, axis=0))
    channels = (
        velocity[0], velocity[1], magnitude, eta, rho, rho_valid,
        confidence, static_valid, dx, dy,
    )
    return np.stack(
        [resize_channel(value, output_hw, i in {5, 7}) for i, value in enumerate(channels)]
    ).astype(np.float32)
