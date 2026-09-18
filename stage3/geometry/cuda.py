"""Batched CUDA geometry with the NumPy pipeline retained as a reference.

Only final feature assembly returns to CPU. Rotation/FOE fitting, expansion,
transport at both lags, and depth-corrected rho share resident CUDA tensors.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np
import torch

from .canonical_grid import canonical_tensor
from .physics_features import physics_vector
from .rotation import rotational_design
from .tracks import _forward_splat_torch


def _quantile(value, q, valid=None):
    if valid is not None:
        value = value.masked_fill(~valid, float('nan'))
    return torch.nanquantile(value, q, dim=-1)


def _fit(design, target, weights):
    # Small normal matrices in float64 avoid the precision loss of normal
    # equations in float32. Pseudoinverse handles stopped/degenerate frames.
    weighted = design * weights[..., None]
    normal = design.transpose(-1, -2) @ weighted
    rhs = weighted.transpose(-1, -2) @ target[..., None]
    return (torch.linalg.pinv(normal, hermitian=True, rtol=1e-12) @ rhs).squeeze(-1)


def rotation_batch(flows, confidence, focal, iterations=3, stride=4):
    n, _, h, w = flows.shape
    design = torch.as_tensor(rotational_design(h, w, focal), device=flows.device)
    a = design[::stride, ::stride].reshape(-1, 3).expand(n, -1, -1)
    b = flows[:, :, ::stride, ::stride].permute(0, 2, 3, 1).reshape(n, -1).double()
    base = confidence[:, ::stride, ::stride].flatten(1).repeat_interleave(2, dim=1).double()
    valid = torch.isfinite(b) & torch.isfinite(base) & (base > .05)
    base = torch.where(valid, base, 0.)
    b = torch.where(valid, b, 0.)
    weights = base
    for _ in range(iterations):
        omega = _fit(a, b, weights)
        residual = b - (a @ omega[..., None]).squeeze(-1)
        scale = (_quantile(residual.abs(), .5, valid) * 1.4826 + 1e-6).nan_to_num(1e-6)
        weights = base / (residual.abs() / (2.5 * scale[:, None])).clamp_min(1)
    omega = torch.where((valid.sum(1) >= 20)[:, None], omega, 0.)
    predicted = torch.einsum('hwkc,nc->nkhw', design, omega).float()
    magnitude = torch.linalg.vector_norm(flows - predicted, dim=1)
    scale = _quantile(magnitude.flatten(1), .5)
    inlier = (magnitude < (2.5 * scale).clamp_min(.25)[:, None, None]).float().mean((1, 2))
    return omega.float(), predicted, inlier, scale


def foe_batch(flows, confidence, iterations=3, stride=4):
    n, _, h, w = flows.shape
    y, x = torch.meshgrid(torch.arange(h, device=flows.device, dtype=torch.float64),
                          torch.arange(w, device=flows.device, dtype=torch.float64), indexing='ij')
    u, v = flows[:, :, ::stride, ::stride].double().unbind(1)
    x, y = x[::stride, ::stride], y[::stride, ::stride]
    magnitude = torch.hypot(u, v)
    confidence = confidence[:, ::stride, ::stride].double()
    valid = (torch.isfinite(magnitude) & (magnitude > .05) & (confidence > .05)).flatten(1)
    a = torch.stack((v, -u), -1).reshape(n, -1, 2)
    b = (v * x - u * y).flatten(1)
    cap = _quantile(magnitude.flatten(1), .9, valid).nan_to_num(1.)
    base = confidence.flatten(1) * torch.minimum(magnitude.flatten(1), cap[:, None])
    base = torch.where(valid, base, 0.)
    a = torch.where(valid[..., None], a, 0.)
    b = torch.where(valid, b, 0.)
    weights = base
    for _ in range(iterations):
        foe = _fit(a, b, weights)
        residual = b - (a @ foe[..., None]).squeeze(-1)
        scale = (_quantile(residual.abs(), .5, valid) * 1.4826 + 1e-6).nan_to_num(1e-6)
        weights = base / (residual.abs() / (2.5 * scale[:, None])).clamp_min(1)
    enough = valid.sum(1) >= 20
    center = flows.new_tensor([(w - 1) / 2, (h - 1) / 2])
    foe = torch.where(enough[:, None], foe, center)
    inlier = ((residual.abs() < 2.5 * scale[:, None]) & valid).sum(1) / valid.sum(1).clamp_min(1)
    return foe.float(), torch.where(enough, inlier, 0.), torch.where(enough, scale, float('inf'))


def transport_lags(fields, flows, lags=(2, 4), batch_size=32):
    """Reuse input tensors and grids across lags; no CPU transfers in the loop."""
    time, h, w = fields.shape
    y, x = torch.meshgrid(torch.arange(h, device=fields.device, dtype=torch.float32),
                          torch.arange(w, device=fields.device, dtype=torch.float32), indexing='ij')
    result = {}
    for lag in lags:
        output, valid_output = torch.zeros_like(fields), torch.zeros_like(fields, dtype=torch.bool)
        for start in range(lag, time, batch_size):
            stop = min(start + batch_size, time)
            tracked = fields[start - lag:stop - lag]
            valid = torch.isfinite(tracked)
            for step in range(1, lag + 1):
                tracked, valid = _forward_splat_torch(tracked, flows[start - lag + step:stop - lag + step], valid, x[None], y[None])
            output[start:stop], valid_output[start:stop] = tracked, valid
        result[lag] = output, valid_output
    return result


def rho_batch(previous, current, delta, tracked_valid):
    delta = delta[:, None, None]
    numerator, denominator = 1 - .5 * delta * previous, 1 + .5 * delta * current
    valid = (tracked_valid & torch.isfinite(previous) & torch.isfinite(current)
             & (previous.abs() > 1e-3) & (current.abs() > 1e-3) & (previous * current > 0)
             & (numerator > 0) & (denominator > 0) & (delta > 0))
    rho = (current.abs().clamp_min(1e-30).log() - previous.abs().clamp_min(1e-30).log()
           + numerator.clamp_min(1e-30).log() - denominator.clamp_min(1e-30).log()) / delta.clamp_min(1e-6)
    return torch.where(valid, rho, 0.), valid


@torch.inference_mode()
def build_cuda_features(flows, confidences, actual_times, focal, output_hw, device, batch_size, calibration_seconds):
    device = torch.device(device)
    if batch_size < 1:
        raise ValueError('geometry.tracking_batch_size must be positive')
    def timestamp():
        torch.cuda.synchronize(device)
        return perf_counter()

    started = timestamp()
    flow = torch.as_tensor(flows, device=device)
    confidence = torch.as_tensor(confidences, device=device)
    transfer_seconds = timestamp() - started
    t, _, h, w = flow.shape
    dt = np.r_[.1, np.maximum(np.diff(actual_times), 1e-3)]
    times = torch.as_tensor(actual_times - dt / 2, device=device)
    dt_tensor = torch.as_tensor(dt, dtype=torch.float32, device=device)
    derotated = torch.empty_like(flow)
    omega = flow.new_empty((t, 3))
    rotation_inlier, rotation_residual = flow.new_empty(t), flow.new_empty(t)
    foes, quality = flow.new_empty((t, 2)), flow.new_empty((t, 2))
    rotation_seconds = foe_seconds = 0.
    for start in range(0, t, batch_size):
        stop = min(start + batch_size, t)
        operation = timestamp()
        om, rotation, ri, rr = rotation_batch(flow[start:stop], confidence[start:stop], focal)
        omega[start:stop], rotation_inlier[start:stop], rotation_residual[start:stop] = om, ri, rr
        derotated[start:stop] = flow[start:stop] - rotation
        rotation_seconds += timestamp() - operation
        operation = timestamp()
        foe, fi, fr = foe_batch(derotated[start:stop], confidence[start:stop])
        foes[start:stop], quality[start:stop, 0], quality[start:stop, 1] = foe, fi, fr
        foe_seconds += timestamp() - operation
    operation = timestamp()
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    dx, dy = x[None] - foes[:, 0, None, None], y[None] - foes[:, 1, None, None]
    eta = ((derotated[:, 0] * dx + derotated[:, 1] * dy)
           / (dx * dx + dy * dy).clamp_min(4) / dt_tensor[:, None, None])
    denominator = 1 + .5 * dt_tensor[:, None, None] * eta
    eta = torch.where(denominator > 1e-6, eta / denominator.clamp_min(1e-6), 0.)
    transport = torch.zeros_like(flow)
    transport[1:] = flow[:-1]
    tracked = transport_lags(eta, transport, batch_size=batch_size)
    rhos = {}
    for lag, (previous, valid) in tracked.items():
        delta = torch.zeros_like(times)
        delta[lag:] = times[lag:] - times[:-lag]
        rhos[lag] = rho_batch(previous, eta, delta.float(), valid)
    tracks_seconds = timestamp() - operation
    operation = timestamp()
    # Download completed fields once, after both lags and rho are finished.
    omega, derotated, eta = omega.cpu().numpy(), derotated.cpu().numpy(), eta.cpu().numpy()
    foes, quality = foes.cpu().numpy(), quality.cpu().numpy()
    rotation_inlier, rotation_residual = rotation_inlier.cpu().numpy(), rotation_residual.cpu().numpy()
    rhos = {lag: (rho.cpu().numpy(), valid.cpu().numpy()) for lag, (rho, valid) in rhos.items()}
    transfer_seconds += timestamp() - operation
    operation = perf_counter()
    motion, physics = np.empty((t, 10, *output_hw), np.float32), np.empty((t, 20), np.float32)
    pitch = omega[:, 0] / dt
    border = max(2, min(h, w) // 50)
    mask = np.ones((h, w), bool)
    mask[:border] = mask[-border:] = False
    mask[:, :border] = mask[:, -border:] = False
    for i in range(t):
        rho2, valid2 = rhos[2][0][i], rhos[2][1][i]
        rho4, valid4 = rhos[4][0][i], rhos[4][1][i]
        static = (mask & (confidences[i] > .15)).astype(np.float32)
        motion[i] = canonical_tensor(derotated[i], dt[i], eta[i], rho2, valid2.astype(np.float32), confidences[i], static, foes[i], output_hw)
        physics[i] = physics_vector(omega[i] / dt[i], rotation_inlier[i], rotation_residual[i],
                                    rho2, valid2, rho4, valid4, eta[i], derotated[i], confidences[i], dt[i],
                                    float(pitch[i] - np.median(pitch[max(0, i - 15):i + 1])))
    feature_seconds = perf_counter() - operation
    if not np.isfinite(motion).all() or not np.isfinite(physics).all():
        raise FloatingPointError('Non-finite values produced by CUDA Stage 3 geometry')
    metadata = {'foe': foes, 'geometry_quality': quality, 'focal_px': np.asarray([focal], np.float32)}
    for name, seconds in [('calibration', calibration_seconds), ('rotation', rotation_seconds), ('foe', foe_seconds),
                          ('tracks_rho', tracks_seconds), ('feature_construction', feature_seconds), ('transfers', transfer_seconds)]:
        metadata['timing_' + name] = np.asarray([seconds])
    return motion, physics, metadata
