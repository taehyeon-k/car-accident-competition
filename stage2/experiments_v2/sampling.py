"""Frame-space temporal sampling. No FPS, duration, or seconds anywhere."""
from __future__ import annotations

import numpy as np


def resample_positions(n_cached: int, count: int, *, jitter: float = 0.0, rng=None) -> np.ndarray:
    """Pick ``count`` cache slots spanning the clip using only ordinal position."""
    if n_cached < 1 or count < 2:
        raise ValueError("cache must be nonempty and count >= 2")
    positions = np.linspace(0.0, n_cached - 1, count)
    if jitter and rng is not None and count > 2:
        positions[1:-1] += rng.uniform(-jitter, jitter, count - 2) * (n_cached - 1) / (count - 1)
        positions = np.maximum.accumulate(np.clip(positions, 0, n_cached - 1))
    return np.rint(positions).astype(np.int64)


def apply_drop(selected: np.ndarray, drop_probability: float, rng) -> np.ndarray:
    """Random frame dropping (v2 section 11). Endpoints are always kept."""
    if not drop_probability or rng is None or len(selected) <= 2:
        return selected
    keep = rng.random(len(selected)) >= drop_probability
    keep[[0, -1]] = True
    return selected[keep]


def apply_warp(selected: np.ndarray, strength: float, rng) -> np.ndarray:
    """Local temporal warping: sample some regions denser than others.

    Reindexes the already-selected slots through a monotone random warp of the
    unit interval, so density varies along the clip while order and endpoints
    are preserved. Purely ordinal - no time units are involved.
    """
    n = len(selected)
    if not strength or rng is None or n <= 3:
        return selected
    knots = np.linspace(0.0, 1.0, 5)
    offsets = np.zeros_like(knots)
    offsets[1:-1] = rng.uniform(-strength, strength, len(knots) - 2)
    warped_knots = np.clip(knots + offsets, 0.0, 1.0)
    warped_knots = np.maximum.accumulate(warped_knots)
    if warped_knots[-1] <= warped_knots[0]:
        return selected
    grid = np.linspace(0.0, 1.0, n)
    warped = np.interp(grid, knots, warped_knots)
    index = np.rint(warped * (n - 1)).astype(np.int64)
    index = np.maximum.accumulate(np.clip(index, 0, n - 1))
    return selected[index]


def apply_duplicate(selected: np.ndarray, probability: float, rng) -> np.ndarray:
    """Optional random frame duplication (v2 section 11)."""
    if not probability or rng is None or len(selected) < 2:
        return selected
    repeats = np.ones(len(selected), dtype=np.int64)
    repeats[rng.random(len(selected)) < probability] = 2
    return np.repeat(selected, repeats)


def build_view(n_cached, count, *, jitter=0.0, drop_probability=0.0, warp=0.0,
               duplicate_probability=0.0, rng=None) -> np.ndarray:
    """One temporally augmented view, expressed as cache slot indices."""
    selected = resample_positions(n_cached, count, jitter=jitter, rng=rng)
    selected = apply_warp(selected, warp, rng)
    selected = apply_drop(selected, drop_probability, rng)
    selected = apply_duplicate(selected, duplicate_probability, rng)
    return selected
