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


def class_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, mode: str, gamma: float = 2.0) -> torch.Tensor:
    """Ordered 3-state auxiliary loss: ``ce``, ``focal`` or cumulative ``ordinal``."""
    target = target.long().clamp(0, 2)
    if mode == "ordinal":
        levels = torch.arange(logits.shape[-1], device=logits.device)
        labels = (target[..., None] > levels).float()
        values = F.binary_cross_entropy_with_logits(logits.float(), labels, reduction="none").sum(-1)
    else:
        log_p = F.log_softmax(logits.float(), -1)
        picked = log_p.gather(-1, target[..., None]).squeeze(-1)
        values = -picked if mode == "ce" else -((1 - picked.exp()) ** gamma) * picked
    return masked_mean(values, mask)


def class_probabilities(logits: torch.Tensor, mode: str) -> torch.Tensor:
    if mode != "ordinal":
        return torch.softmax(logits.float(), -1)
    above = torch.sigmoid(logits.float())
    above = torch.cummin(above, -1).values
    ones, zeros = torch.ones_like(above[..., :1]), torch.zeros_like(above[..., :1])
    cumulative = torch.cat((ones, above, zeros), -1)
    return (cumulative[..., :-1] - cumulative[..., 1:]).clamp_min(0)


def stage3_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], cfg: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Final-stage loss plus, when refinement is enabled, down-weighted earlier stages."""
    total, parts = _stage_loss(outputs, batch, cfg)
    level = 1
    while f"s{level}_acceleration" in outputs:
        prefix = f"s{level}_"
        stage = {key[len(prefix):]: value for key, value in outputs.items() if key.startswith(prefix)}
        stage_total, stage_parts = _stage_loss(stage, batch, cfg)
        total = total + float(cfg.get("refinement_stage_weight", 0.5)) * stage_total
        parts[f"stage{level}_total"] = stage_total.detach()
        level += 1
    return total, parts


def _stage_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], cfg: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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
    parts = {}
    if "jerk" in outputs:
        parts["jerk"] = masked_huber(outputs["jerk"], batch["jerk"], time & batch["valid_jerk"].bool())
    if "steer_rate" in outputs:
        parts["steer_rate"] = masked_huber(outputs["steer_rate"], batch["steer_rate"],
                                           time & batch["valid_steer_rate"].bool(), cfg.get("steer_rate_delta", 10.0))
    mode = cfg.get("class_loss", "ce")
    if "accel_class" in outputs:
        mean = 0.5 * (batch["a_long_s1"] + batch["a_long_s2"])
        label = (mean > cfg.get("accelerating_above", 0.25)).long() * 2 + ((mean >= cfg.get("decelerating_below", -0.25)) & (mean <= cfg.get("accelerating_above", 0.25))).long()
        mask = accel_mask & torch.isfinite(mean) & (batch["stopped"] < 0.5)
        parts["accel_class"] = class_loss(outputs["accel_class"], label, mask, mode, cfg.get("focal_gamma", 2.0))
    if "steer_class" in outputs:
        angle = batch["steering_angle"]
        threshold = cfg.get("steering_threshold_deg", 5.0)
        # Ordered RIGHT(0) < STRAIGHT(1) < LEFT(2), positive angle = LEFT.
        label = (angle > threshold).long() * 2 + ((angle >= -threshold) & (angle <= threshold)).long()
        mask = steer_mask & torch.isfinite(angle)
        parts["steer_class"] = class_loss(outputs["steer_class"], label, mask, mode, cfg.get("focal_gamma", 2.0))
    parts.update({
        "accel_direct": direct, "ordinal": ordinal, "stopped": stopped,
        "speed": speed, "accel_speed_consistency": consistency,
        "steering_angle": steering, "yaw_aux": yaw,
    })
    weights = cfg["weights"]
    total = sum(float(weights.get(name, 0.0)) * value for name, value in parts.items())
    return total, {name: value.detach() for name, value in parts.items()}
