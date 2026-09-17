from __future__ import annotations

import numpy as np


def forward_splat(values: np.ndarray, flow: np.ndarray, valid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Advect a scalar field along forward flow with bilinear splatting."""
    h, w = values.shape
    y, x = np.mgrid[:h, :w]
    destination_x = x + flow[0]
    destination_y = y + flow[1]
    base_valid = np.isfinite(values) & np.isfinite(destination_x) & np.isfinite(destination_y)
    if valid is not None:
        base_valid &= valid
    total = np.zeros((h, w), np.float64)
    weight_total = np.zeros((h, w), np.float64)
    for xi, yi, weight in (
        (np.floor(destination_x), np.floor(destination_y), (1 - destination_x % 1) * (1 - destination_y % 1)),
        (np.ceil(destination_x), np.floor(destination_y), (destination_x % 1) * (1 - destination_y % 1)),
        (np.floor(destination_x), np.ceil(destination_y), (1 - destination_x % 1) * (destination_y % 1)),
        (np.ceil(destination_x), np.ceil(destination_y), (destination_x % 1) * (destination_y % 1)),
    ):
        xi, yi = xi.astype(np.int64), yi.astype(np.int64)
        keep = base_valid & (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h) & (weight > 0)
        np.add.at(total, (yi[keep], xi[keep]), values[keep] * weight[keep])
        np.add.at(weight_total, (yi[keep], xi[keep]), weight[keep])
    output_valid = weight_total > 1e-5
    output = np.zeros((h, w), np.float32)
    output[output_valid] = (total[output_valid] / weight_total[output_valid]).astype(np.float32)
    return output, output_valid


def advect_scalar(values: np.ndarray, forward_flows: list[np.ndarray], valid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    tracked, tracked_valid = values, np.isfinite(values) if valid is None else valid.copy()
    for flow in forward_flows:
        tracked, tracked_valid = forward_splat(tracked, flow, tracked_valid)
    return tracked, tracked_valid
