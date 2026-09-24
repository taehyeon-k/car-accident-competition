from __future__ import annotations

import numpy as np


def normalized_indices(frame_numbers, count: int, *, jitter: float = 0.0, rng=None):
    """Select normalized positions using only ordered original frame numbers."""
    frames = np.asarray(frame_numbers, dtype=np.int64)
    if frames.ndim != 1 or not len(frames) or count < 2:
        raise ValueError("frame_numbers must be nonempty and count >= 2")
    if (np.diff(frames) <= 0).any():
        raise ValueError("frame numbers must be strictly increasing")
    positions = np.linspace(0.0, 1.0, count)
    if jitter and rng is not None and count > 2:
        positions[1:-1] += rng.uniform(-jitter, jitter, count - 2) / (count - 1)
        positions = np.maximum.accumulate(np.clip(positions, 0.0, 1.0))
    targets = frames[0] + positions * (frames[-1] - frames[0])
    upper = np.searchsorted(frames, targets).clip(0, len(frames) - 1)
    lower = np.maximum(upper - 1, 0)
    return np.where(targets - frames[lower] <= frames[upper] - targets, lower, upper).astype(np.int64)


def sample_from_cache(frame_numbers, count: int, *, jitter=0.0, drop_probability=0.0, rng=None):
    # An FPS-blind base grid can repeat original frames when a clip has fewer
    # frames than cache positions. Resample the normalized cache axis itself;
    # requiring unique frame numbers here would reject valid short clips.
    frames = np.asarray(frame_numbers)
    if frames.ndim != 1 or not len(frames) or (np.diff(frames) < 0).any():
        raise ValueError("cached frame numbers must be nonempty and nondecreasing")
    positions = np.linspace(0.0, len(frames) - 1, count)
    if jitter and rng is not None and count > 2:
        positions[1:-1] += rng.uniform(-jitter, jitter, count - 2) * (len(frames) - 1) / (count - 1)
        positions = np.maximum.accumulate(np.clip(positions, 0, len(frames) - 1))
    selected = np.rint(positions).astype(np.int64)
    if drop_probability and rng is not None and len(selected) > 2:
        keep = rng.random(len(selected)) >= drop_probability
        keep[[0, -1]] = True
        selected = selected[keep]
    return selected
