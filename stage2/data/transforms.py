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


def letterbox_resize(
    image: torch.Tensor,
    size: int,
) -> torch.Tensor:
    """Resize a CHW RGB image to fit inside ``size`` square, still in [0, 1].

    Split out from :func:`letterbox` so clip augmentation can run at backbone
    resolution instead of native resolution, and so it never touches the padding.
    """
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Expected a CHW RGB image with exactly three channels")
    _, original_height, original_width = image.shape
    if size <= 0 or original_width <= 0 or original_height <= 0:
        raise ValueError("Image and target dimensions must be positive")
    scale = min(size / original_width, size / original_height)
    return TVF.resize(
        _as_unit_float(image),
        size=[
            max(1, round(original_height * scale)),
            max(1, round(original_width * scale)),
        ],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )


def letterbox_pad_normalize(
    resized: torch.Tensor,
    size: int,
    original_width: int,
    original_height: int,
) -> tuple[torch.Tensor, Letterbox]:
    """Mean-colour pad a resized image to square, then ImageNet-normalize it.

    A custom canvas remains necessary because torchvision's tensor padding takes
    a scalar fill while this architecture requires a three-channel ImageNet-mean
    fill. Padding before Normalize is what makes the border exactly zero, so the
    padding must never be augmented.
    """
    resized_height, resized_width = resized.shape[-2:]
    mean_colour = torch.tensor(
        IMAGENET_MEAN,
        dtype=resized.dtype,
        device=resized.device,
    )
    canvas = mean_colour[:, None, None].expand(3, size, size).clone()
    pad_x = (size - resized_width) // 2
    pad_y = (size - resized_height) // 2
    canvas[:, pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
    normalized = TVF.normalize(canvas, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return normalized, letterbox_metadata(original_width, original_height, size)


def letterbox(
    image: torch.Tensor,
    size: int,
) -> tuple[torch.Tensor, Letterbox]:
    """Resize, mean-colour letterbox, and ImageNet-normalize one CHW RGB image."""
    _, original_height, original_width = image.shape
    resized = letterbox_resize(image, size)
    return letterbox_pad_normalize(resized, size, original_width, original_height)


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
