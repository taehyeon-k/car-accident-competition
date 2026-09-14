"""Clip-consistent online augmentation for the joint Stage 2 training path.

Every parameter is sampled **once per clip** and then applied identically to all
of its frames, so DINOv3 and V-JEPA always observe the same pixels and no
frame-to-frame photometric flicker is introduced. Geometry follows the same
decision: the horizontal flip transforms the cached detector boxes before
tracking, so ``center_x`` and ``dx`` mirror without a second correction.

Only appearance is altered. No spatial resampling (crop, rotation, perspective,
affine, vertical flip) and no colour-destroying transform (solarize, posterize,
grayscale, strong hue) is available here by construction: such transforms would
move the entering vehicle, change traffic geometry, or stop looking like dashcam
footage.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TVF

COLOUR_TRANSFORMS = ("brightness", "contrast", "gamma", "saturation")

DEFAULTS = {
    "horizontal_flip": {"probability": 0.5},
    "photometric": {
        "brightness": {"probability": 0.35, "factor_min": 0.75, "factor_max": 1.25},
        "contrast": {"probability": 0.35, "factor_min": 0.80, "factor_max": 1.20},
        "gamma": {"probability": 0.25, "gamma_min": 0.85, "gamma_max": 1.15},
        "saturation": {"probability": 0.25, "factor_min": 0.80, "factor_max": 1.20},
        "jpeg": {"probability": 0.25, "quality_min": 60, "quality_max": 95},
        "gaussian_blur": {
            "probability": 0.10,
            "kernel_size": 5,
            "sigma_min": 0.10,
            "sigma_max": 1.20,
        },
        "gaussian_noise": {"probability": 0.15, "std_min": 0.005, "std_max": 0.025},
    },
}


def _merge(defaults: dict, override: dict | None) -> dict:
    merged = {k: dict(v) if isinstance(v, dict) else v for k, v in defaults.items()}
    for key, value in (override or {}).items():
        if key not in merged:
            raise ValueError(f"Unknown augmentation option {key!r}")
        if isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def augmentation_config(config: dict | None) -> dict:
    """Fill unset augmentation fields from the documented defaults."""
    return _merge(DEFAULTS, config)


@dataclass(frozen=True)
class ClipPhotometric:
    """One clip's frozen photometric decision, reused by every frame."""

    order: tuple[str, ...] = ()
    brightness: float | None = None
    contrast: float | None = None
    gamma: float | None = None
    saturation: float | None = None
    jpeg_quality: int | None = None
    blur: tuple[int, float] | None = None
    noise_std: float | None = None

    @property
    def active(self) -> tuple[str, ...]:
        names = tuple(n for n in self.order if getattr(self, n) is not None)
        extra = tuple(
            name
            for name, value in (
                ("jpeg", self.jpeg_quality),
                ("blur", self.blur),
                ("noise", self.noise_std),
            )
            if value is not None
        )
        return names + extra

    def __call__(self, image: torch.Tensor, generator=None) -> torch.Tensor:
        """Apply this clip's configuration to one CHW RGB image in [0, 1]."""
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("Expected a CHW RGB image with three channels")
        out = image
        for name in self.order:
            value = getattr(self, name)
            if value is None:
                continue
            if name == "brightness":
                out = TVF.adjust_brightness(out, value)
            elif name == "contrast":
                out = TVF.adjust_contrast(out, value)
            elif name == "gamma":
                out = TVF.adjust_gamma(out.clamp(0, 1), value)
            else:
                out = TVF.adjust_saturation(out, value)
            out = out.clamp(0, 1)
        # Sensor-side degradations always follow the colour/intensity stage.
        if self.jpeg_quality is not None:
            out = _jpeg(out, self.jpeg_quality)
        if self.blur is not None:
            kernel, sigma = self.blur
            out = TVF.gaussian_blur(out, [kernel, kernel], [sigma, sigma])
        if self.noise_std is not None:
            # Strength is fixed for the clip; the pixel pattern may vary per frame.
            noise = torch.empty_like(out).normal_(
                0.0, self.noise_std, generator=generator
            )
            out = out + noise
        return out.clamp(0, 1)


def _jpeg(image: torch.Tensor, quality: int) -> torch.Tensor:
    buffer = io.BytesIO()
    array = (image.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
    Image.fromarray(array).save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    decoded = np.asarray(Image.open(buffer).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(decoded).permute(2, 0, 1).to(image.dtype)


def sample_photometric(config: dict, rng: np.random.Generator) -> ClipPhotometric:
    """Draw one clip-level configuration. Called once per training sample."""
    photometric = config["photometric"]

    def enabled(name: str) -> bool:
        return bool(rng.random() < photometric[name]["probability"])

    def uniform(name: str, low: str, high: str) -> float:
        section = photometric[name]
        return float(rng.uniform(section[low], section[high]))

    # The colour order is drawn once and then frozen for the whole clip.
    order = list(COLOUR_TRANSFORMS)
    rng.shuffle(order)
    blur = None
    if enabled("gaussian_blur"):
        blur = (
            int(photometric["gaussian_blur"]["kernel_size"]),
            uniform("gaussian_blur", "sigma_min", "sigma_max"),
        )
    return ClipPhotometric(
        order=tuple(order),
        brightness=(
            uniform("brightness", "factor_min", "factor_max")
            if enabled("brightness")
            else None
        ),
        contrast=(
            uniform("contrast", "factor_min", "factor_max")
            if enabled("contrast")
            else None
        ),
        gamma=uniform("gamma", "gamma_min", "gamma_max") if enabled("gamma") else None,
        saturation=(
            uniform("saturation", "factor_min", "factor_max")
            if enabled("saturation")
            else None
        ),
        jpeg_quality=(
            int(
                rng.integers(
                    photometric["jpeg"]["quality_min"],
                    photometric["jpeg"]["quality_max"] + 1,
                )
            )
            if enabled("jpeg")
            else None
        ),
        blur=blur,
        noise_std=(
            uniform("gaussian_noise", "std_min", "std_max")
            if enabled("gaussian_noise")
            else None
        ),
    )


def sample_horizontal_flip(config: dict, rng: np.random.Generator) -> bool:
    return bool(rng.random() < config["horizontal_flip"]["probability"])


def flip_boxes(boxes: torch.Tensor, width: int) -> torch.Tensor:
    """Mirror native xyxy boxes: [x1, y1, x2, y2] -> [W - x2, y1, W - x1, y2]."""
    if boxes.numel() == 0:
        return boxes.clone()
    if boxes.shape[-1] != 4:
        raise ValueError("Expected native xyxy boxes")
    mirrored = boxes.clone().to(dtype=torch.float32)
    x1 = mirrored[..., 0].clone()
    mirrored[..., 0] = width - mirrored[..., 2]
    mirrored[..., 2] = width - x1
    return mirrored


def flip_side(side) -> str | int:
    """LEFT <-> RIGHT for either the string or the encoded integer label."""
    if isinstance(side, str):
        if side not in ("LEFT", "RIGHT"):
            raise ValueError(f"Unknown entry_side {side!r}")
        return "RIGHT" if side == "LEFT" else "LEFT"
    if side not in (0, 1):
        raise ValueError(f"Unknown encoded entry_side {side!r}")
    return 1 - int(side)
