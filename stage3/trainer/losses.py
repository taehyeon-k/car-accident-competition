from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool() & torch.isfinite(values)
    if not mask.any():
        return values.masked_select(mask).sum()
    return values.masked_select(mask).mean()


def masked_huber(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    values = F.huber_loss(prediction.float(), target.float().nan_to_num(), reduction="none", delta=delta)
    return masked_mean(values, mask & torch.isfinite(target))


def tied_ordinal_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Random thresholds supervise the continuous latent without a class head."""
    count = int(cfg.get("samples", 8))
    low, high = map(float, cfg.get("range", [-3.0, 3.0]))
    temperature = float(cfg.get("temperature", 0.25))
    thresholds = torch.empty((*prediction.shape, count), device=prediction.device).uniform_(low, high)
    logits = (prediction[..., None].float() - thresholds) / temperature
    labels = (target[..., None] > thresholds).float()
    values = F.binary_cross_entropy_with_logits(logits, labels, reduction="none").mean(-1)
    return masked_mean(values, mask & torch.isfinite(target))


def stage3_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], cfg: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    time = batch["time_valid"].bool()
    accel_mask = time & batch["valid_accel"].bool()
    derivative_mask = time & batch["valid_accel_speed"].bool()
    speed_mask = time & batch["valid_speed"].bool()
    steer_mask = time & batch["valid_steer"].bool()
    yaw_mask = time & batch["valid_yaw"].bool()
    accel = outputs["acceleration"]
    direct = 0.5 * (
        masked_huber(accel[..., 0], batch["a_long_s1"], accel_mask)
        + masked_huber(accel[..., 1], batch["a_long_s2"], accel_mask)
    )
    consistency = 0.5 * (
        masked_huber(accel[..., 0], batch["a_dvdt_s1"], derivative_mask)
        + masked_huber(accel[..., 1], batch["a_dvdt_s2"], derivative_mask)
    )
    ordinal = 0.5 * (
        tied_ordinal_loss(accel[..., 0], batch["a_long_s1"], accel_mask, cfg["ordinal"])
        + tied_ordinal_loss(accel[..., 1], batch["a_long_s2"], accel_mask, cfg["ordinal"])
    )
    stopped = masked_mean(
        F.binary_cross_entropy_with_logits(outputs["stop_logit"].float(), batch["stopped"].float(), reduction="none"),
        speed_mask & torch.isfinite(batch["stopped"]),
    )
    speed = masked_huber(outputs["speed"], batch["speed"], speed_mask)
    steering = masked_huber(outputs["steering_angle"], batch["steering_angle"], steer_mask, cfg.get("steering_delta", 2.0))
    yaw = outputs["speed"].new_zeros(())
    if "yaw_rate_aux" in outputs:
        yaw = masked_huber(outputs["yaw_rate_aux"], batch["yaw_rate_aux"], yaw_mask)
    parts = {
        "accel_direct": direct, "ordinal": ordinal, "stopped": stopped,
        "speed": speed, "accel_speed_consistency": consistency,
        "steering_angle": steering, "yaw_aux": yaw,
    }
    weights = cfg["weights"]
    total = sum(float(weights.get(name, 0.0)) * value for name, value in parts.items())
    return total, {name: value.detach() for name, value in parts.items()}
