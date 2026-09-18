"""Select a fixed grid of relative clip positions without FPS or duration metadata."""
import numpy as np


def normalized_indices(frame_numbers, count=128, rng=None):
    values = np.asarray(frame_numbers, dtype=np.float64)
    if values.ndim != 1 or not len(values) or count < 2:
        raise ValueError('Expected nonempty frame numbers and at least two sampling positions')
    if not np.isfinite(values).all() or (np.diff(values) <= 0).any():
        raise ValueError('Frame numbers must be finite and strictly increasing')
    positions = np.linspace(0, 1, count)
    if rng is not None:
        # Cadence jitter during training, retaining endpoints and temporal order.
        positions[1:-1] += rng.uniform(-0.45, 0.45, count-2)/(count-1)
    targets = values[0] + positions*(values[-1]-values[0])
    upper = np.searchsorted(values, targets).clip(0, len(values)-1)
    lower = np.maximum(upper-1, 0)
    return np.where(targets-values[lower] <= values[upper]-targets, lower, upper).astype(np.int64)
