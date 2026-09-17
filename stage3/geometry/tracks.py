from __future__ import annotations

import numpy as np
import torch


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


def _forward_splat_torch(
    values: torch.Tensor,
    flow: torch.Tensor,
    valid: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched equivalent of :func:`forward_splat` using scatter-add."""
    batch, height, width = values.shape
    destination_x = x + flow[:, 0]
    destination_y = y + flow[:, 1]
    x0, y0 = torch.floor(destination_x), torch.floor(destination_y)
    fx, fy = destination_x - x0, destination_y - y0
    base_valid = valid & torch.isfinite(values) & torch.isfinite(destination_x) & torch.isfinite(destination_y)
    total = torch.zeros((batch, height * width), dtype=values.dtype, device=values.device)
    weight_total = torch.zeros_like(total)

    for xi, yi, weight in (
        (x0, y0, (1 - fx) * (1 - fy)),
        (x0 + 1, y0, fx * (1 - fy)),
        (x0, y0 + 1, (1 - fx) * fy),
        (x0 + 1, y0 + 1, fx * fy),
    ):
        keep = base_valid & (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height) & (weight > 0)
        safe_x = torch.where(keep, xi, torch.zeros_like(xi)).clamp(0, width - 1).long()
        safe_y = torch.where(keep, yi, torch.zeros_like(yi)).clamp(0, height - 1).long()
        flat_index = (safe_y * width + safe_x).flatten(1)
        kept_weight = torch.where(keep, weight, torch.zeros_like(weight)).flatten(1)
        total.scatter_add_(1, flat_index, (values * kept_weight.view_as(values)).flatten(1))
        weight_total.scatter_add_(1, flat_index, kept_weight)

    output_valid = weight_total > 1e-5
    output = torch.where(output_valid, total / weight_total.clamp_min(1e-5), torch.zeros_like(total))
    return output.view(batch, height, width), output_valid.view(batch, height, width)


@torch.inference_mode()
def advect_lagged_fields(
    values: np.ndarray,
    forward_flows: np.ndarray,
    lag: int,
    device: str | torch.device = "cuda",
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Track every ``values[t-lag]`` to ``t`` in batched Torch operations.

    Flow index ``j`` maps frame ``j-1`` to frame ``j``. The returned arrays have
    the same time/height/width shape as ``values``; the first ``lag`` entries are
    zero and invalid.
    """
    fields = np.asarray(values, dtype=np.float32)
    flows = np.asarray(forward_flows, dtype=np.float32)
    if fields.ndim != 3 or flows.shape != (len(fields), 2, *fields.shape[1:]):
        raise ValueError("Expected values [T,H,W] and forward_flows [T,2,H,W]")
    if lag < 1 or batch_size < 1:
        raise ValueError("lag and batch_size must be positive")

    time, height, width = fields.shape
    tracked_output = np.zeros_like(fields)
    valid_output = np.zeros_like(fields, dtype=bool)
    if time <= lag:
        return tracked_output, valid_output

    torch_device = torch.device(device)
    field_tensor = torch.from_numpy(fields).to(torch_device)
    flow_tensor = torch.from_numpy(flows).to(torch_device)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, device=torch_device, dtype=torch.float32),
        torch.arange(width, device=torch_device, dtype=torch.float32),
        indexing="ij",
    )
    grid_x, grid_y = grid_x.unsqueeze(0), grid_y.unsqueeze(0)

    for current_start in range(lag, time, batch_size):
        current_stop = min(time, current_start + batch_size)
        tracked = field_tensor[current_start - lag : current_stop - lag]
        tracked_valid = torch.isfinite(tracked)
        for step in range(1, lag + 1):
            step_flow = flow_tensor[current_start - lag + step : current_stop - lag + step]
            tracked, tracked_valid = _forward_splat_torch(
                tracked, step_flow, tracked_valid, grid_x, grid_y
            )
        tracked_output[current_start:current_stop] = tracked.cpu().numpy()
        valid_output[current_start:current_stop] = tracked_valid.cpu().numpy()
    return tracked_output, valid_output
