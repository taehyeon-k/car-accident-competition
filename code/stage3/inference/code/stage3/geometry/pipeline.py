from __future__ import annotations

import numpy as np
from time import perf_counter

from .calibration import estimate_focal
from .canonical_grid import canonical_tensor
from .foe import estimate_foe
from .physics_features import physics_vector
from .rho import radial_expansion, rho_from_expansion
from .rotation import estimate_rotation
from .tracks import advect_lagged_fields, advect_scalar


def build_motion_features(
    flows: np.ndarray,
    confidences: np.ndarray,
    actual_times: np.ndarray,
    calibration_cfg: dict,
    output_hw: tuple[int, int] = (96, 168),
    calibration_frame: np.ndarray | None = None,
    tracking_device: str | None = None,
    tracking_batch_size: int = 32,
    geometry_backend: str = "auto",
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Convert forward flows into canonical 10-channel maps and 20-D physics."""
    if geometry_backend not in {"auto", "numpy"}:
        raise ValueError("geometry_backend must be auto or numpy")
    t, _, h, w = flows.shape
    motion = np.zeros((t, 10, *output_hw), np.float32)
    physics = np.zeros((t, 20), np.float32)
    calibration_start = perf_counter()
    focal = estimate_focal(w, h, calibration_cfg, calibration_frame)
    calibration_seconds = perf_counter() - calibration_start
    if tracking_device is not None:
        import torch
        device = torch.device(tracking_device)
        if geometry_backend == "auto" and device.type == "cuda" and torch.cuda.is_available():
            from .cuda import build_cuda_features
            return build_cuda_features(flows, confidences, actual_times, focal, output_hw, device,
                                       tracking_batch_size, calibration_seconds)
    expansions, foes = [], []
    intervals = np.r_[0.1, np.maximum(np.diff(actual_times), 1e-3)]
    expansion_times = actual_times - intervals / 2
    rotations, derotated_all, quality = [], [], []
    rotation_seconds = 0.0
    foe_seconds = 0.0
    for i in range(t):
        dt = 0.1 if i == 0 else max(float(actual_times[i] - actual_times[i - 1]), 1e-3)
        operation_start = perf_counter()
        omega, rotation_flow, rot_inlier, rot_residual = estimate_rotation(flows[i], confidences[i], focal)
        derotated = flows[i] - rotation_flow
        rotation_seconds += perf_counter() - operation_start
        operation_start = perf_counter()
        foe, foe_inlier, foe_residual = estimate_foe(derotated, confidences[i])
        foe_seconds += perf_counter() - operation_start
        eta = radial_expansion(derotated, foe, dt)
        # Forward displacement / source radius is delta_Z / Z_end.
        # Convert to interval-midpoint expansion before comparing speed ratios.
        denominator = 1 + 0.5 * dt * eta
        eta = np.divide(eta, denominator, out=np.zeros_like(eta), where=denominator > 1e-6)
        expansions.append(eta)
        foes.append(foe)
        rotations.append((omega, rot_inlier, rot_residual))
        derotated_all.append(derotated)
        quality.append((foe_inlier, foe_residual))
    transport_flows = np.zeros_like(flows)
    transport_flows[1:] = flows[:-1]
    tracks_rho_seconds = 0.0
    feature_seconds = 0.0
    pitch_rates = []
    for i, (omega, _, _) in enumerate(rotations):
        dt = 0.1 if i == 0 else max(float(actual_times[i] - actual_times[i - 1]), 1e-3)
        pitch_rates.append(float(omega[0] / dt))
    tracked_by_lag: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    batched_tracking_start = perf_counter()
    if tracking_device is not None:
        import torch

        requested_device = torch.device(tracking_device)
        if requested_device.type != "cuda" or torch.cuda.is_available():
            expansion_array = np.asarray(expansions, dtype=np.float32)
            flow_array = transport_flows
            for lag in (2, 4):
                tracked_by_lag[lag] = advect_lagged_fields(
                    expansion_array, flow_array, lag, requested_device, tracking_batch_size
                )
    tracks_rho_seconds += perf_counter() - batched_tracking_start
    for i in range(t):
        dt = 0.1 if i == 0 else max(float(actual_times[i] - actual_times[i - 1]), 1e-3)
        zero = np.zeros((h, w), np.float32)
        false = np.zeros((h, w), bool)
        operation_start = perf_counter()
        if i >= 2:
            if 2 in tracked_by_lag:
                tracked2, track_valid2 = tracked_by_lag[2][0][i], tracked_by_lag[2][1][i]
            else:
                tracked2, track_valid2 = advect_scalar(expansions[i - 2], transport_flows[i - 1 : i + 1])
            rho2, valid2 = rho_from_expansion(tracked2, expansions[i], expansion_times[i] - expansion_times[i - 2])
            valid2 &= track_valid2
        else:
            rho2, valid2 = zero, false
        if i >= 4:
            if 4 in tracked_by_lag:
                tracked4, track_valid4 = tracked_by_lag[4][0][i], tracked_by_lag[4][1][i]
            else:
                tracked4, track_valid4 = advect_scalar(expansions[i - 4], transport_flows[i - 3 : i + 1])
            rho4, valid4 = rho_from_expansion(tracked4, expansions[i], expansion_times[i] - expansion_times[i - 4])
            valid4 &= track_valid4
        else:
            rho4, valid4 = zero, false
        tracks_rho_seconds += perf_counter() - operation_start
        operation_start = perf_counter()
        border = max(2, min(h, w) // 50)
        camera_mask = np.ones((h, w), bool)
        camera_mask[:border] = camera_mask[-border:] = False
        camera_mask[:, :border] = camera_mask[:, -border:] = False
        static_valid = (camera_mask & (confidences[i] > 0.15)).astype(np.float32)
        motion[i] = canonical_tensor(
            derotated_all[i], dt, expansions[i], rho2, valid2.astype(np.float32),
            confidences[i], static_valid, foes[i], output_hw,
        )
        omega, rot_inlier, rot_residual = rotations[i]
        pitch_baseline = float(np.median(pitch_rates[max(0, i - 15) : i + 1]))
        pitch_proxy = pitch_rates[i] - pitch_baseline
        physics[i] = physics_vector(
            omega / dt, rot_inlier, rot_residual, rho2, valid2, rho4, valid4,
            expansions[i], derotated_all[i], confidences[i], dt, pitch_proxy,
        )
        feature_seconds += perf_counter() - operation_start
    metadata = {
        "foe": np.asarray(foes),
        "geometry_quality": np.asarray(quality),
        "focal_px": np.asarray([focal], np.float32),
        "timing_calibration": np.asarray([calibration_seconds]),
        "timing_rotation": np.asarray([rotation_seconds]),
        "timing_foe": np.asarray([foe_seconds]),
        "timing_tracks_rho": np.asarray([tracks_rho_seconds]),
        "timing_feature_construction": np.asarray([feature_seconds]),
    }
    if not np.isfinite(motion).all() or not np.isfinite(physics).all():
        raise FloatingPointError("Non-finite values produced by Stage 3 geometry")
    return motion, physics, metadata
