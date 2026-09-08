"""Spatial preprocessing shared by V-JEPA and DINO branches.

Torchvision performs interpolation and ImageNet normalization. The small custom
letterbox wrapper only supplies the per-image coordinate metadata required to map
native detector boxes into the backbone patch grid.
"""

from __future__ import annotations

from dataclasses import dataclass
import io

import numpy as np
from PIL import Image

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TVF

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Letterbox:
    """The exact native-image to square-image transform for one frame."""

    scale: float
    pad_x: float
    pad_y: float
    width: int
    height: int
    size: int
    scale_x: float | None = None
    scale_y: float | None = None


def letterbox_metadata(
    width: int,
    height: int,
    size: int,
) -> Letterbox:
    """Compute exactly the same integer resize and padding without decoding RGB."""
    if (
        min(
            width,
            height,
            size,
        )
        <= 0
    ):
        raise ValueError("Image and target dimensions must be positive")
    scale = min(
        size / width,
        size / height,
    )
    resized_width = max(
        1,
        round(width * scale),
    )
    resized_height = max(
        1,
        round(height * scale),
    )
    return Letterbox(
        scale=scale,
        pad_x=(size - resized_width) // 2,
        pad_y=(size - resized_height) // 2,
        width=width,
        height=height,
        size=size,
        scale_x=resized_width / width,
        scale_y=resized_height / height,
    )


def _as_unit_float(image: torch.Tensor) -> torch.Tensor:
    """Use torchvision conversion so uint8 images are correctly scaled to [0, 1]."""
    if image.dtype == torch.uint8:
        return TVF.convert_image_dtype(
            image,
            torch.float32,
        )
    if not image.is_floating_point():
        raise ValueError("RGB input must be uint8 or floating-point [0,1]")
    return image.to(dtype=torch.float32)


def letterbox(
    image: torch.Tensor,
    size: int,
) -> tuple[torch.Tensor, Letterbox]:
    """Resize, mean-colour letterbox, and ImageNet-normalize one CHW RGB image.

    ``torchvision.transforms.functional.resize`` is used instead of a handwritten
    interpolation call. A custom canvas remains necessary because torchvision's
    tensor padding accepts a scalar fill, while this architecture requires a
    three-channel ImageNet-mean fill. It also preserves the exact pad offsets for
    detector-box to ROIAlign-coordinate conversion.
    """
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Expected a CHW RGB image with exactly three channels")

    _, original_height, original_width = image.shape
    if size <= 0 or original_width <= 0 or original_height <= 0:
        raise ValueError("Image and target dimensions must be positive")
    scale = min(
        size / original_width,
        size / original_height,
    )
    resized_width = max(
        1,
        round(original_width * scale),
    )
    resized_height = max(
        1,
        round(original_height * scale),
    )

    unit_float = _as_unit_float(image)
    resized = TVF.resize(
        unit_float,
        size=[resized_height, resized_width],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )

    # Letterbox before Normalize: after Normalize this mean-colour padding is zero.
    mean_colour = torch.tensor(
        IMAGENET_MEAN,
        dtype=resized.dtype,
        device=resized.device,
    )
    canvas = (
        mean_colour[:, None, None]
        .expand(
            3,
            size,
            size,
        )
        .clone()
    )
    pad_x = (size - resized_width) // 2
    pad_y = (size - resized_height) // 2
    canvas[:, pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized

    normalized = TVF.normalize(
        canvas,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    )
    transform = letterbox_metadata(
        original_width,
        original_height,
        size,
    )
    return normalized, transform


def box_to_grid(
    box: torch.Tensor,
    transform: Letterbox,
    patch_size: int,
) -> torch.Tensor:
    """Map an ``xyxy`` native-coordinate box to V-JEPA/DINO patch-grid coordinates."""
    if patch_size <= 0 or box.shape[-1] != 4:
        raise ValueError("Expected xyxy boxes and a positive patch size")
    # Integer resize rounding can give different exact x/y scale factors.
    scale_x = transform.scale_x if transform.scale_x is not None else transform.scale
    scale_y = transform.scale_y if transform.scale_y is not None else transform.scale
    grid_box = box.to(dtype=torch.float32).clone()
    grid_box[..., 0::2] = (grid_box[..., 0::2] * scale_x + transform.pad_x) / patch_size
    grid_box[..., 1::2] = (grid_box[..., 1::2] * scale_y + transform.pad_y) / patch_size
    return grid_box


class ClipPhotometric:
    """Sample once per clip/window and reuse torchvision transform parameters.

    Noise uses the same sampled spatial pattern for every frame to avoid flicker.
    The specification leaves degradation strengths open; these defaults are mild.
    Detector/depth cache creation must never call this transform.
    """

    def __init__(
        self,
        rng: np.random.Generator,
    ) -> None:
        self.brightness = (
            float(
                rng.uniform(
                    0.70,
                    1.30,
                )
            )
            if rng.random() < 0.5
            else 1.0
        )
        self.contrast = (
            float(
                rng.uniform(
                    0.80,
                    1.20,
                )
            )
            if rng.random() < 0.5
            else 1.0
        )
        self.saturation = (
            float(
                rng.uniform(
                    0.80,
                    1.20,
                )
            )
            if rng.random() < 0.3
            else 1.0
        )
        self.gamma = (
            float(
                rng.uniform(
                    0.80,
                    1.20,
                )
            )
            if rng.random() < 0.3
            else 1.0
        )
        self.hue = (
            float(
                rng.uniform(
                    -0.03,
                    0.03,
                )
            )
            if rng.random() < 0.2
            else 0.0
        )
        self.noise_std = (
            float(
                rng.uniform(
                    0.002,
                    0.01,
                )
            )
            if rng.random() < 0.2
            else 0.0
        )
        self.jpeg_quality = (
            int(
                rng.integers(
                    65,
                    96,
                )
            )
            if rng.random() < 0.2
            else None
        )
        self.blur_sigma = (
            float(
                rng.uniform(
                    0.1,
                    0.8,
                )
            )
            if rng.random() < 0.1
            else None
        )
        self.noise_seed = int(rng.integers(2**31))

    def __call__(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """Transform a CPU CHW RGB image without moving pixels geometrically."""
        image = _as_unit_float(image)
        image = TVF.adjust_brightness(
            image,
            self.brightness,
        )
        image = TVF.adjust_contrast(
            image,
            self.contrast,
        )
        image = TVF.adjust_saturation(
            image,
            self.saturation,
        )
        image = TVF.adjust_gamma(
            image,
            self.gamma,
        )
        image = TVF.adjust_hue(
            image,
            self.hue,
        )
        if self.noise_std:
            generator = torch.Generator(device=image.device).manual_seed(
                self.noise_seed
            )
            noise = torch.randn(
                image.shape,
                generator=generator,
                device=image.device,
            )
            image = (image + self.noise_std * noise).clamp(
                0,
                1,
            )
        if self.jpeg_quality is not None:
            buffer = io.BytesIO()
            TVF.to_pil_image(image).save(
                buffer,
                format="JPEG",
                quality=self.jpeg_quality,
            )
            buffer.seek(0)
            with Image.open(buffer) as compressed:
                image = TVF.to_tensor(compressed.convert("RGB"))
        if self.blur_sigma is not None:
            image = TVF.gaussian_blur(
                image,
                kernel_size=[3, 3],
                sigma=[self.blur_sigma, self.blur_sigma],
            )
        return image
