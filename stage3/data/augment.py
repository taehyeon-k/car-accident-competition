from __future__ import annotations

import torch


def horizontal_flip(motion: torch.Tensor, physics: torch.Tensor, targets: dict[str, torch.Tensor] | None = None):
    """Mirror canonical features and all lateral quantities consistently."""
    result_motion = torch.flip(motion, dims=(-1,)).clone()
    result_motion[..., 0, :, :] *= -1  # flow x
    result_motion[..., 8, :, :] *= -1  # FOE-centered dx
    result_physics = physics.clone()
    result_physics[..., 1:3] *= -1  # mirror axial-vector omega_y/z
    result_targets = None if targets is None else {key: value.clone() for key, value in targets.items()}
    if result_targets is not None:
        for key in ("steering_angle", "yaw_rate_aux"):
            if key in result_targets:
                result_targets[key] *= -1
    return result_motion, result_physics, result_targets
