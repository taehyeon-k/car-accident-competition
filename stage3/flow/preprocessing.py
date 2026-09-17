from __future__ import annotations

import numpy as np


def validate_rgb_frames(frames: list[np.ndarray]) -> tuple[int, int]:
    if not frames:
        raise ValueError("At least one RGB frame is required")
    shape = frames[0].shape
    if len(shape) != 3 or shape[2] != 3 or any(frame.shape != shape for frame in frames):
        raise ValueError("Flow frames must have a common HxWx3 shape")
    return shape[:2]
