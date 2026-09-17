from __future__ import annotations

import numpy as np
from time import perf_counter

from .calibration import estimate_focal
from .canonical_grid import canonical_tensor
from .foe import estimate_foe
from .physics_features import physics_vector
from .rho import radial_expansion, rho_from_expansion
from .rotation import estimate_rotation
from .tracks import advect_scalar


def build_motion_features(
    flows: np.ndarray,
    confidences: np.ndarray,
    actual_times: np.ndarray,
    calibration_cfg: dict,
    output_hw: tuple[int, int] = (96, 168),
    calibration_frame: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Convert forward flows into canonical 10-channel maps and 20-D physics."""
    t, _, h, w = flows.shape
    motion = np.zeros((t, 10, *output_hw), np.float32)
    physics = np.zeros((t, 20), np.float32)
    calibration_start = perf_counter()
    focal = estimate_focal(w, h, calibration_cfg, calibration_frame)
    calibration_seconds = perf_counter() - calibration_start
    expansions, foes = [], []
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
        expansions.append(eta)
        foes.append(foe)
        rotations.append((omega, rot_inlier, rot_residual))
        derotated_all.append(derotated)
        quality.append((foe_inlier, foe_residual))
    tracks_rho_seconds = 0.0
    feature_seconds = 0.0
    pitch_rates = []
    for i, (omega, _, _) in enumerate(rotations):
        dt = 0.1 if i == 0 else max(float(actual_times[i] - actual_times[i - 1]), 1e-3)
        pitch_rates.append(float(omega[0] / dt))
    for i in range(t):
        dt = 0.1 if i == 0 else max(float(actual_times[i] - actual_times[i - 1]), 1e-3)
        zero = np.zeros((h, w), np.float32)
        false = np.zeros((h, w), bool)
        operation_start = perf_counter()
        if i >= 2:
            tracked2, track_valid2 = advect_scalar(expansions[i - 2], derotated_all[i - 1 : i + 1])
            rho2, valid2 = rho_from_expansion(tracked2, expansions[i], actual_times[i] - actual_times[i - 2])
            valid2 &= track_valid2
        else:
            rho2, valid2 = zero, false
        if i >= 4:
            tracked4, track_valid4 = advect_scalar(expansions[i - 4], derotated_all[i - 3 : i + 1])
            rho4, valid4 = rho_from_expansion(tracked4, expansions[i], actual_times[i] - actual_times[i - 4])
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
