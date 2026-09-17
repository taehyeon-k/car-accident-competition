from __future__ import annotations

import re
import sys
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    convnext_tiny,
    efficientnet_b0,
    resnet18,
)


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Cache-build constants (must match src/stage1_rebuild/extract_png.py defaults used
# to build the training frames manifest: every video cached 8 sparse + 2x16 burst).
SPARSE_FRAMES = 8
GLOBAL_FRAME_INDICES = (0, 2, 5, 7)
BURST_COUNT = 2
BURST_FRAMES = 16
cv2.setNumThreads(1)


# --------------------------------------------------------------------------- #
# Model (stage1_rebuild lineage: RGB backbone + native-FPS burst statistics)
# --------------------------------------------------------------------------- #
class AttentionPool(nn.Module):
    def __init__(self, feature_dim: int, axis: int):
        super().__init__()
        self.axis = axis
        self.score = nn.Sequential(
            nn.Linear(feature_dim, max(16, feature_dim // 2)),
            nn.Tanh(),
            nn.Linear(max(16, feature_dim // 2), 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(features), dim=self.axis)
        return torch.sum(features * weights, dim=self.axis)


class ImageBackbone(nn.Module):
    def __init__(self, name: str, pretrained: bool):
        super().__init__()
        if name == "efficientnet_b0":
            model = efficientnet_b0(
                weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None
            )
            self.encoder = model.features
            self.output_dim = model.classifier[-1].in_features
        elif name == "resnet18":
            model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
            self.encoder = nn.Sequential(*list(model.children())[:-2])
            self.output_dim = model.fc.in_features
        elif name == "convnext_tiny":
            model = convnext_tiny(
                weights=ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            )
            self.encoder = model.features
            self.output_dim = model.classifier[-1].in_features
        else:
            raise ValueError(f"unsupported backbone: {name}")
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.pool(self.forward_map(images)).flatten(1)

    def forward_map(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)


class TemporalBlock(nn.Module):
    def __init__(self, feature_dim: int, dilation: int, dropout: float):
        super().__init__()
        self.convolutions = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=dilation, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=dilation, dilation=dilation),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded = self.convolutions(features.transpose(1, 2)).transpose(1, 2)
        return self.norm(features + encoded)


class BurstEncoder(nn.Module):
    def __init__(self, input_channels: int = 9, output_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.profile_encoder = nn.Sequential(
            nn.Conv1d(input_channels, 32, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, output_dim),
            nn.GELU(),
        )
        self.temporal = nn.Sequential(
            TemporalBlock(output_dim, 1, dropout),
            TemporalBlock(output_dim, 2, dropout),
        )
        self.frame_pool = AttentionPool(output_dim, axis=1)
        self.burst_pool = AttentionPool(output_dim, axis=1)

    def forward(self, profiles: torch.Tensor) -> torch.Tensor:
        batch, burst_count, time, channels, bins = profiles.shape
        encoded = self.profile_encoder(
            profiles.reshape(batch * burst_count * time, channels, bins)
        ).reshape(batch * burst_count, time, -1)
        encoded = self.temporal(encoded)
        bursts = self.frame_pool(encoded).reshape(batch, burst_count, -1)
        return self.burst_pool(bursts)


class GatedLateFusion(nn.Module):
    def __init__(
        self,
        sparse_dim: int,
        global_dim: int,
        burst_dim: int,
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.sparse_projection = nn.Linear(sparse_dim, output_dim)
        self.global_projection = nn.Linear(global_dim, output_dim)
        self.burst_projection = nn.Linear(burst_dim, output_dim)
        self.gate = nn.Linear(sparse_dim + global_dim + burst_dim, 3)
        self.output = nn.Sequential(nn.LayerNorm(output_dim), nn.GELU(), nn.Dropout(dropout))

    def forward(
        self, sparse: torch.Tensor, global_feature: torch.Tensor, burst: torch.Tensor
    ) -> torch.Tensor:
        weights = torch.softmax(
            self.gate(torch.cat([sparse, global_feature, burst], dim=-1)), dim=-1
        )
        fused = (
            weights[:, 0:1] * self.sparse_projection(sparse)
            + weights[:, 1:2] * self.global_projection(global_feature)
            + weights[:, 2:3] * self.burst_projection(burst)
        )
        return self.output(fused)


class Stage1RebuildModel(nn.Module):
    """Local RGB patches + global frames + native-FPS burst statistics."""

    def __init__(
        self,
        backbone: str = "convnext_tiny",
        pretrained: bool = False,
        sparse_dim: int = 256,
        burst_dim: int = 128,
        burst_input_channels: int = 9,
        residual_input_channels: int = 0,
        residual_scale_init: float = 0.1,
        residual_additive_enabled: bool = True,
        dropout: float = 0.2,
        modality_dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.rgb_backbone = ImageBackbone(backbone, pretrained)
        rgb_dim = self.rgb_backbone.output_dim
        self.rgb_patch_pool = AttentionPool(rgb_dim, axis=2)
        self.sparse_projection = nn.Sequential(
            nn.Linear(rgb_dim, sparse_dim),
            nn.LayerNorm(sparse_dim),
            nn.GELU(),
        )
        self.sparse_temporal = nn.Sequential(
            TemporalBlock(sparse_dim, 1, dropout),
            TemporalBlock(sparse_dim, 2, dropout),
        )
        self.sparse_pool = AttentionPool(sparse_dim, axis=1)
        self.global_spatial_pool = nn.AdaptiveAvgPool2d((2, 5))
        self.global_projection = nn.Sequential(
            nn.Linear(rgb_dim * 10, sparse_dim),
            nn.LayerNorm(sparse_dim),
            nn.GELU(),
        )
        self.global_temporal = TemporalBlock(sparse_dim, 1, dropout)
        self.global_pool = AttentionPool(sparse_dim, axis=1)
        self.burst_input_channels = burst_input_channels
        self.residual_input_channels = residual_input_channels
        self.residual_additive_enabled = residual_additive_enabled
        self.burst_encoder = BurstEncoder(
            input_channels=burst_input_channels, output_dim=burst_dim, dropout=dropout
        )
        if residual_input_channels:
            self.residual_encoder = BurstEncoder(
                input_channels=residual_input_channels,
                output_dim=burst_dim,
                dropout=dropout,
            )
            self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        else:
            self.residual_encoder = None
            self.register_parameter("residual_scale", None)
        self.modality_dropout = modality_dropout
        self.fusion = GatedLateFusion(
            sparse_dim=sparse_dim,
            global_dim=sparse_dim,
            burst_dim=burst_dim,
            output_dim=sparse_dim,
            dropout=dropout,
        )
        self.classifier = nn.Linear(sparse_dim, 1)
        self.sparse_auxiliary = nn.Linear(sparse_dim, 1)
        self.global_auxiliary = nn.Linear(sparse_dim, 1)
        self.burst_auxiliary = nn.Linear(burst_dim, 1)
        self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(
        self,
        patches: torch.Tensor,
        global_frames: torch.Tensor,
        burst_profiles: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, time, patch_count, channels, height, width = patches.shape
        flat = patches.reshape(batch * time * patch_count, channels, height, width)
        normalized = (flat - self.rgb_mean) / self.rgb_std
        rgb = self.rgb_backbone(normalized).reshape(batch, time, patch_count, -1)
        rgb = self.rgb_patch_pool(rgb)
        sparse_sequence = self.sparse_projection(rgb)
        sparse = self.sparse_pool(self.sparse_temporal(sparse_sequence))

        global_batch, global_time, channels, global_height, global_width = global_frames.shape
        if global_batch != batch:
            raise ValueError("local and global batch sizes differ")
        global_flat = global_frames.reshape(
            global_batch * global_time, channels, global_height, global_width
        )
        global_normalized = (global_flat - self.rgb_mean) / self.rgb_std
        global_map = self.rgb_backbone.forward_map(global_normalized)
        global_sequence = self.global_projection(
            self.global_spatial_pool(global_map).flatten(1)
        ).reshape(batch, global_time, -1)
        global_feature = self.global_pool(self.global_temporal(global_sequence))

        burst = self.burst_encoder(
            burst_profiles[:, :, :, : self.burst_input_channels, :]
        )
        if self.residual_encoder is not None:
            residual = self.residual_encoder(
                burst_profiles[:, :, :, self.burst_input_channels :, :]
            )
            if self.residual_additive_enabled:
                burst = burst + self.residual_scale * residual
        fused = self.fusion(sparse, global_feature, burst)
        return {
            "logit": self.classifier(fused).squeeze(-1),
            "sparse_logit": self.sparse_auxiliary(sparse).squeeze(-1),
            "global_logit": self.global_auxiliary(global_feature).squeeze(-1),
            "burst_logit": self.burst_auxiliary(burst).squeeze(-1),
        }


# --------------------------------------------------------------------------- #
# Stage 1 prediction
# --------------------------------------------------------------------------- #
def predict_stage1(data_dir, model_dir):
    data_path = _resolve_stage_dir(Path(data_dir), "stage1")
    model_path = _resolve_stage_dir(Path(model_dir), "stage1")
    checkpoint_path = _find_checkpoint(model_path)
    device = _choose_device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {}) or {}
    residual_profiles = bool(config.get("residual_profiles", False))
    independent_residual_encoder = bool(config.get("independent_residual_encoder", False))
    model = Stage1RebuildModel(
        backbone=str(config.get("backbone", "convnext_tiny")),
        burst_input_channels=(
            9 if independent_residual_encoder or not residual_profiles else 14
        ),
        residual_input_channels=5 if independent_residual_encoder else 0,
        residual_scale_init=float(config.get("residual_scale_init", 0.1)),
        residual_additive_enabled=bool(
            config.get("residual_additive_enabled", True)
        ),
        dropout=float(config.get("dropout", 0.2)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    threshold = float(checkpoint.get("threshold", 0.5))
    image_size = int(config.get("image_size", 224))
    center_ratio = float(config.get("center_ratio", 0.8))
    patch_ratio = float(config.get("patch_ratio", 0.30))
    row_bins = int(config.get("row_bins", 64))
    global_height = int(config.get("global_height", 192))
    global_width = int(config.get("global_width", 320))
    residual_sigma_fine = float(config.get("residual_sigma_fine", 1.0))
    residual_sigma_coarse = float(config.get("residual_sigma_coarse", 2.0))
    residual_profile_scales = tuple(
        float(value)
        for value in config.get("residual_profile_scales", [1.0] * 5)
    )

    rows = []
    for video_id, video_path in _discover_video_items(data_path):
        patches, global_frames, burst_profiles = _prepare_video_inputs(
            video_path,
            image_size=image_size,
            center_ratio=center_ratio,
            patch_ratio=patch_ratio,
            row_bins=row_bins,
            global_height=global_height,
            global_width=global_width,
            residual_profiles=residual_profiles,
            residual_sigma_fine=residual_sigma_fine,
            residual_sigma_coarse=residual_sigma_coarse,
            residual_profile_scales=residual_profile_scales,
        )
        with torch.no_grad(), torch.autocast(
            device_type=device.type, enabled=device.type == "cuda"
        ):
            probability = float(
                torch.sigmoid(
                    model(
                        patches.to(device),
                        global_frames.to(device),
                        burst_profiles.to(device),
                    )["logit"]
                ).mean().cpu()
            )
        rows.append(
            {
                "ID": video_id,
                "answer": "RERECORDED" if probability >= threshold else "ORIGINAL",
            }
        )
    return pd.DataFrame(rows, columns=["ID", "answer"])


def _prepare_video_inputs(
    path: Path,
    *,
    image_size: int,
    center_ratio: float,
    patch_ratio: float,
    row_bins: int,
    global_height: int,
    global_width: int,
    residual_profiles: bool,
    residual_sigma_fine: float,
    residual_sigma_coarse: float,
    residual_profile_scales: tuple[float, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    frame_count = _probe_frame_count(path)
    sparse_indices, burst_starts = _select_indices(frame_count)
    wanted = set(int(np.clip(i, 0, frame_count - 1)) for i in sparse_indices)
    for start in burst_starts:
        for offset in range(BURST_FRAMES):
            wanted.add(int(np.clip(start + offset, 0, frame_count - 1)))
    frames_by_index = _collect_frames(path, wanted)

    sparse_frames = [frames_by_index[int(np.clip(i, 0, frame_count - 1))] for i in sparse_indices]
    patches = np.stack([
        _extract_relative_patches(rgb, image_size, center_ratio, patch_ratio)
        for rgb in sparse_frames
    ]).astype(np.float32) / 255.0
    patch_tensor = torch.from_numpy(patches).permute(0, 1, 4, 2, 3).contiguous().unsqueeze(0)

    global_frames = np.stack(
        [
            cv2.resize(
                sparse_frames[index],
                (global_width, global_height),
                interpolation=(
                    cv2.INTER_AREA
                    if sparse_frames[index].shape[0] > global_height
                    or sparse_frames[index].shape[1] > global_width
                    else cv2.INTER_CUBIC
                ),
            )
            for index in GLOBAL_FRAME_INDICES
        ]
    ).astype(np.float32) / 255.0
    global_tensor = (
        torch.from_numpy(global_frames).permute(0, 3, 1, 2).contiguous().unsqueeze(0)
    )

    burst_profiles = []
    for start in burst_starts:
        group = [
            frames_by_index[int(np.clip(start + offset, 0, frame_count - 1))]
            for offset in range(BURST_FRAMES)
        ]
        burst_profiles.append(
            _temporal_profiles(
                group,
                center_ratio,
                row_bins,
                residual_profiles=residual_profiles,
                residual_sigma_fine=residual_sigma_fine,
                residual_sigma_coarse=residual_sigma_coarse,
                residual_profile_scales=residual_profile_scales,
            )
        )
    burst_tensor = torch.from_numpy(np.stack(burst_profiles).astype(np.float32)).unsqueeze(0)
    return patch_tensor, global_tensor, burst_tensor


def _select_indices(frame_count: int) -> tuple[list[int], list[int]]:
    frame_count = max(frame_count, 2)
    sparse_margin = min(max(1, round(frame_count * 0.05)), (frame_count - 1) // 2)
    sparse_indices = (
        np.linspace(sparse_margin, frame_count - 1 - sparse_margin, SPARSE_FRAMES)
        .round()
        .astype(int)
        .tolist()
    )
    maximum_start = max(0, frame_count - BURST_FRAMES)
    burst_starts = (
        np.linspace(0, maximum_start, BURST_COUNT + 2)[1:-1].round().astype(int).tolist()
    )
    return sparse_indices, burst_starts


def _collect_frames(path: Path, wanted: set[int]) -> dict[int, np.ndarray]:
    """Decode sequentially, keeping only the requested frame indices. Missing
    trailing indices (short video / bad metadata) fall back to the last frame."""
    maximum = max(wanted)
    frames: dict[int, np.ndarray] = {}
    last = None
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        index = -1
        for frame in container.decode(stream):
            index += 1
            last = frame.to_ndarray(format="rgb24")
            if index in wanted:
                frames[index] = last
            if index >= maximum:
                break
    if last is None:
        raise RuntimeError(f"failed to decode any frame from {path}")
    for target in wanted:
        frames.setdefault(target, last)
    return frames


def _probe_frame_count(path: Path) -> int:
    # Mirror src/stage1_rebuild/extract_png.py, which selects frames by index over
    # frame_count = round(fps * duration). Matching this exactly is what keeps the
    # sampled frame indices identical to the cached training frames (zero skew).
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or stream.base_rate or stream.guessed_rate or 0.0)
        duration = (
            float(stream.duration * stream.time_base)
            if stream.duration is not None and stream.time_base is not None
            else float(container.duration / av.time_base) if container.duration else 0.0
        )
        if fps > 0 and duration > 0:
            return max(1, int(round(fps * duration)))
        if stream.frames and stream.frames > 0:
            return int(stream.frames)
    return _video_frame_count(path)


def _extract_relative_patches(
    rgb: np.ndarray, image_size: int, center_ratio: float, patch_ratio: float
) -> np.ndarray:
    height, width = rgb.shape[:2]
    roi_width = max(2, round(width * center_ratio))
    roi_height = max(2, round(height * center_ratio))
    x0 = (width - roi_width) // 2
    y0 = (height - roi_height) // 2
    patch_size = max(2, round(min(width, height) * patch_ratio))
    patch_size = min(patch_size, roi_width, roi_height)
    centers_x = (x0 + roi_width / 3, x0 + 2 * roi_width / 3)
    centers_y = (y0 + roi_height / 3, y0 + 2 * roi_height / 3)
    result = []
    for center_y in centers_y:
        for center_x in centers_x:
            x = round(center_x - patch_size / 2)
            y = round(center_y - patch_size / 2)
            x = min(max(x0, x), x0 + roi_width - patch_size)
            y = min(max(y0, y), y0 + roi_height - patch_size)
            crop = rgb[y : y + patch_size, x : x + patch_size]
            result.append(
                cv2.resize(
                    crop,
                    (image_size, image_size),
                    interpolation=cv2.INTER_AREA if patch_size > image_size else cv2.INTER_CUBIC,
                )
            )
    return np.stack(result)


def _temporal_profiles(
    frames: list[np.ndarray],
    center_ratio: float,
    row_bins: int,
    *,
    residual_profiles: bool,
    residual_sigma_fine: float,
    residual_sigma_coarse: float,
    residual_profile_scales: tuple[float, ...],
) -> np.ndarray:
    row_profiles = []
    column_profiles = []
    global_means = []
    luminances = []
    for rgb in frames:
        height, width = rgb.shape[:2]
        roi_width = max(2, round(width * center_ratio))
        roi_height = max(2, round(height * center_ratio))
        x0 = (width - roi_width) // 2
        y0 = (height - roi_height) // 2
        roi = rgb[y0 : y0 + roi_height, x0 : x0 + roi_width]
        luminance = (0.299 * roi[:, :, 0] + 0.587 * roi[:, :, 1] + 0.114 * roi[:, :, 2]) / 255.0
        luminances.append(luminance)
        row_profiles.append(
            cv2.resize(luminance, (1, row_bins), interpolation=cv2.INTER_AREA).reshape(-1)
        )
        column_profiles.append(
            cv2.resize(luminance, (row_bins, 1), interpolation=cv2.INTER_AREA).reshape(-1)
        )
        global_means.append(float(np.mean(luminance)))
    row = np.stack(row_profiles)
    column = np.stack(column_profiles)
    row_gradient = np.diff(row, axis=1, prepend=row[:, :1])
    column_gradient = np.diff(column, axis=1, prepend=column[:, :1])
    temporal_row = np.diff(row, axis=0, prepend=row[:1])
    temporal_column = np.diff(column, axis=0, prepend=column[:1])
    difference_energy = [0.0]
    for previous, current in zip(luminances, luminances[1:]):
        if previous.shape != current.shape:
            current = cv2.resize(
                current, (previous.shape[1], previous.shape[0]), interpolation=cv2.INTER_AREA
            )
        difference_energy.append(float(np.mean(np.abs(current - previous))))
    difference = np.repeat(
        np.asarray(difference_energy, dtype=np.float32)[:, None], row_bins, axis=1
    )
    duplicate = (difference < (0.5 / 255.0)).astype(np.float32)
    global_profile = np.repeat(
        np.asarray(global_means, dtype=np.float32)[:, None], row_bins, axis=1
    )
    base_profiles = np.stack(
        [
            row,
            row_gradient,
            column,
            column_gradient,
            temporal_row,
            temporal_column,
            global_profile,
            difference,
            duplicate,
        ],
        axis=1,
    ).astype(np.float32)
    if not residual_profiles:
        return base_profiles
    residual = _residual_profiles_from_luminances(
        luminances,
        row_bins=row_bins,
        sigma_fine=residual_sigma_fine,
        sigma_coarse=residual_sigma_coarse,
    )
    scales = np.asarray(residual_profile_scales, dtype=np.float32)[None, :, None]
    return np.concatenate(
        [base_profiles, np.clip(residual / scales, 0.0, 1.0)], axis=1
    ).astype(np.float32)


def _residual_profiles_from_luminances(
    luminances: list[np.ndarray],
    *,
    row_bins: int,
    sigma_fine: float,
    sigma_coarse: float,
) -> np.ndarray:
    fine_rows = []
    fine_columns = []
    mid_rows = []
    mid_columns = []
    for luminance in luminances:
        fine_blurred = cv2.GaussianBlur(
            luminance, (0, 0), sigmaX=sigma_fine, sigmaY=sigma_fine
        )
        coarse_blurred = cv2.GaussianBlur(
            luminance, (0, 0), sigmaX=sigma_coarse, sigmaY=sigma_coarse
        )
        fine = np.abs(luminance - fine_blurred)
        mid = np.abs(fine_blurred - coarse_blurred)
        fine_rows.append(
            cv2.resize(fine, (1, row_bins), interpolation=cv2.INTER_AREA).reshape(-1)
        )
        fine_columns.append(
            cv2.resize(fine, (row_bins, 1), interpolation=cv2.INTER_AREA).reshape(-1)
        )
        mid_rows.append(
            cv2.resize(mid, (1, row_bins), interpolation=cv2.INTER_AREA).reshape(-1)
        )
        mid_columns.append(
            cv2.resize(mid, (row_bins, 1), interpolation=cv2.INTER_AREA).reshape(-1)
        )
    fine_row = np.stack(fine_rows)
    fine_column = np.stack(fine_columns)
    temporal_fine_change = np.zeros_like(fine_row)
    temporal_fine_change[1:] = 0.5 * (
        np.abs(np.diff(fine_row, axis=0))
        + np.abs(np.diff(fine_column, axis=0))
    )
    return np.stack(
        [
            fine_row,
            fine_column,
            np.stack(mid_rows),
            np.stack(mid_columns),
            temporal_fine_change,
        ],
        axis=1,
    ).astype(np.float32)


# --------------------------------------------------------------------------- #
# Stage 2: joint DINOv3 + V-JEPA (LoRA) model, decoded exactly as in validation
# --------------------------------------------------------------------------- #
import os
import sys
import time

# Validation decodes with COLLISION - ENTRY <= 200 frames (validation.max_span_frames).
STAGE2_MAX_SPAN_FRAMES = 200
# Stage 2 must leave time for Stages 1 and 3 inside the 60-minute limit. Videos
# still pending when the budget runs out get the placeholder prediction instead of
# risking a timeout that would void the whole submission.
STAGE2_TIME_BUDGET_S = float(os.environ.get("STAGE2_TIME_BUDGET_S", 40 * 60))
STAGE2_COLUMNS = ["ID", "collision_frame", "entry_frame", "evasion_space", "entry_side"]
# Inference-only execution settings: they change speed and memory, not the model.
# Swept on the longest validation clip (1,279 frames, L40S) with output-drift checks:
#   detector 4  - larger batches were at most 10% faster but changed box counts on
#                 up to 204 frames, so 4 keeps detections identical to the training cache
#   DINO 8      - fastest measured; 16-256 were slower with no benefit
#   V-JEPA 1    - clip batch 2 saved 0.6 s but moved global features by up to 4.6e-2
#   no activation checkpointing - 7% faster V-JEPA, zero drift (it only saves memory
#                 when gradients are kept, and inference keeps none)
STAGE2_DETECTOR_BATCH = int(os.environ.get("STAGE2_DETECTOR_BATCH", 4))
STAGE2_DINO_BATCH = int(os.environ.get("STAGE2_DINO_BATCH", 8))
STAGE2_VJEPA_CLIP_BATCH = int(os.environ.get("STAGE2_VJEPA_CLIP_BATCH", 1))
STAGE2_VJEPA_ACTIVATION_CHECKPOINTING = (
    os.environ.get("STAGE2_VJEPA_ACTIVATION_CHECKPOINTING", "0") == "1"
)
# Event decoding, chosen on validation (outputs/decode_study):
#   "window" - sum each frame's predicted probability over the +/-0.3 s window the
#              metric tolerates, then pick the ordered pair maximizing the weighted
#              sum, with COLLISION - ENTRY <= 200 frames. Test fps is unknown, so the
#              radius comes from clip length (train/val: 50 frames = 10 fps, 150 =
#              15 fps, 500+ = ~30 fps), which assumes test clips follow that pattern.
#   "span"   - highest joint logit with COLLISION - ENTRY <= 200 frames.
STAGE2_DECODER = os.environ.get("STAGE2_DECODER", "window")
STAGE2_WINDOW_WEIGHT = float(os.environ.get("STAGE2_WINDOW_WEIGHT", 0.5))


def predict_stage2(data_dir, model_dir):
    started = time.monotonic()
    data_path = _resolve_stage_dir(Path(data_dir), "stage2")
    items = _discover_stage2_items(data_path)
    device = _choose_device()
    try:
        system, detector, config = _load_stage2(Path(model_dir), device)
    except Exception as error:  # a failed load must still yield a valid submission
        print(f"[stage2] model load failed ({error!r}); using placeholders", flush=True)
        return pd.DataFrame(
            [_stage2_placeholder(item_id, path) for item_id, path in items],
            columns=STAGE2_COLUMNS,
        )
    rows = []
    for item_id, item_path in items:
        if time.monotonic() - started > STAGE2_TIME_BUDGET_S:
            print(f"[stage2] time budget reached; placeholder for {item_id}", flush=True)
            rows.append(_stage2_placeholder(item_id, item_path))
            continue
        try:
            rows.append(_predict_stage2_item(system, detector, config, item_id, item_path, device))
        except Exception as error:
            print(f"[stage2] {item_id} failed ({error!r}); using placeholder", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            rows.append(_stage2_placeholder(item_id, item_path))
    return pd.DataFrame(rows, columns=STAGE2_COLUMNS)


def _load_stage2(model_dir: Path, device: torch.device):
    model_dir = model_dir.expanduser().resolve()
    stage_dir = model_dir if (model_dir / "joint_model.pt").is_file() else model_dir / "stage2"
    # Backbone sources and RF-DETR weights ship inside the package; the factories
    # read this location when stage2.model.local_assets is first imported.
    os.environ["STAGE2_PRETRAINED"] = str(stage_dir / "pretrained")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    code_dir = str(stage_dir / "code")
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    from stage2.model.backbones import FrozenAdapter, resolve_factory
    from stage2.model.joint_system import JointSystem

    checkpoint = torch.load(stage_dir / "joint_model.pt", map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model_config = dict(config["model"])
    model_config.update(
        dino_batch_size=STAGE2_DINO_BATCH,
        vjepa_clip_batch_size=STAGE2_VJEPA_CLIP_BATCH,
        vjepa_activation_checkpointing=STAGE2_VJEPA_ACTIVATION_CHECKPOINTING,
    )
    # Architectures come from source only: every weight, frozen or trained, is in
    # joint_model.pt, so the multi-GB pretrained checkpoints are not needed.
    system = JointSystem(
        model_config,
        encoder=resolve_factory(model_config["vjepa_factory"])(),
        dino=resolve_factory(model_config["dino_factory"])(),
    )
    system.load_state_dict(checkpoint["model"], strict=True)
    detector = FrozenAdapter(
        "stage2.model.local_assets:detector",
        str(stage_dir / "pretrained" / "rfdetr_small" / "rf-detr-small.pth"),
    )
    return system.eval().to(device), detector.to(device), config


def _predict_stage2_item(system, detector, config, item_id, item_path, device):
    from stage2.data.cache_geometry import compact_observations, frame_paths
    from stage2.data.joint import joint_collate, joint_item
    from torchvision.io import ImageReadMode, read_image

    tracking = config.get("tracking", {})
    threshold = float(tracking.get("detection_threshold", 0.2))
    batch_size = STAGE2_DETECTOR_BATCH
    max_span = (config.get("validation") or {}).get("max_span_frames", STAGE2_MAX_SPAN_FRAMES)

    paths, frame_ids = frame_paths(str(item_path))
    # Same frozen detector, threshold and batch size that built the training cache.
    records = []
    for start in range(0, len(paths), batch_size):
        images = [
            read_image(str(path), mode=ImageReadMode.RGB).to(device)
            for path in paths[start : start + batch_size]
        ]
        for image, detected in zip(images, detector(images)):
            records.append(
                compact_observations(detected, image.shape[-1], image.shape[-2], threshold)
            )
    # Complete video, no augmentation, crop-local tracking: the validation path.
    item = joint_item({"sample_id": item_id}, paths, frame_ids, records, 0, len(paths), tracking=tracking)
    batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in joint_collate([item]).items()
    }
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        output = system(batch)
    entry, collision = _decode_stage2(
        output["entry_logits"], output["collision_logits"], len(paths), max_span
    )
    return {
        "ID": item_id,
        "collision_frame": int(frame_ids[collision]),
        "entry_frame": int(frame_ids[entry]),
        "evasion_space": int(float(output["evasion_logits"][0]) >= 0),
        "entry_side": "LEFT" if int(output["side_logits"][0].float().argmax()) == 0 else "RIGHT",
    }


def _decode_stage2(entry_logits, collision_logits, frame_count, max_span):
    """Return (entry_index, collision_index) from one video's [1, T] event logits."""
    from stage2.utils.joint_losses import constrained_decode

    if STAGE2_DECODER == "span":
        entry, collision = constrained_decode(entry_logits, collision_logits, max_span)
        return int(entry[0]), int(collision[0])
    if STAGE2_DECODER != "window":
        raise ValueError(f"unknown STAGE2_DECODER {STAGE2_DECODER!r}")
    fps = 10.0 if frame_count <= 80 else 15.0 if frame_count <= 400 else 30.0
    radius = int(0.3 * fps + 1e-6)
    scores = []
    for logits, weight in (
        (entry_logits, STAGE2_WINDOW_WEIGHT),
        (collision_logits, 1 - STAGE2_WINDOW_WEIGHT),
    ):
        # CPU float32, exactly as validated, so GPU rounding cannot flip a near-tie.
        probability = torch.softmax(logits[0].float().cpu(), -1)
        cumulative = torch.cat([probability.new_zeros(1), probability.cumsum(0)])
        frames = torch.arange(len(probability))
        window = (
            cumulative[(frames + radius + 1).clamp(max=len(probability))]
            - cumulative[(frames - radius).clamp(min=0)]
        )
        # 1e-7 x probability only separates exact plateau ties, toward the peak.
        scores.append(weight * window + 1e-7 * probability)
    entry, collision = constrained_decode(scores[0][None], scores[1][None], max_span)
    return int(entry[0]), int(collision[0])


def _stage2_placeholder(item_id, item_path):
    """The uploaded file's original Stage 2 heuristic, kept as a safe fallback."""
    first_frame, last_frame = _stage2_frame_bounds(item_path)
    span = max(1, last_frame - first_frame)
    entry_frame = first_frame + round(span * 0.50)
    collision_frame = first_frame + round(span * 0.75)
    if collision_frame <= entry_frame:
        collision_frame = min(last_frame, entry_frame + 1)
    return {
        "ID": item_id,
        "collision_frame": int(collision_frame),
        "entry_frame": int(entry_frame),
        "evasion_space": 1,
        "entry_side": "LEFT",
    }


# --------------------------------------------------------------------------- #
# Stage 3: acceleration-centric SEA-RAFT + motion TCN
# --------------------------------------------------------------------------- #
def predict_stage3(data_dir, model_dir):
    model_path = Path(model_dir).expanduser().resolve()
    code_root = model_path / "stage3" / "code"
    if code_root.is_dir() and str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    from stage3.inference.dacon import predict_stage3 as stage3_predict

    return stage3_predict(data_dir, model_dir)


# --------------------------------------------------------------------------- #
# Shared discovery / IO helpers (unchanged)
# --------------------------------------------------------------------------- #
def _discover_video_items(data_path: Path) -> list[tuple[str, Path]]:
    labels_path = data_path / "labels.csv"
    if labels_path.is_file():
        labels = pd.read_csv(labels_path, dtype={"ID": str, "path": str})
        if {"ID", "path"}.issubset(labels.columns):
            return [
                (str(row.ID), (data_path / row.path).resolve())
                for row in labels.itertuples()
            ]
    videos_root = data_path / "videos"
    search_root = videos_root if videos_root.is_dir() else data_path
    paths = sorted(
        path
        for path in search_root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    return [(path.stem, path.resolve()) for path in paths]


def _discover_stage2_items(data_path: Path) -> list[tuple[str, Path]]:
    labels_path = data_path / "labels.csv"
    if labels_path.is_file():
        labels = pd.read_csv(labels_path, dtype={"ID": str, "path": str})
        if {"ID", "path"}.issubset(labels.columns):
            return [
                (str(row.ID), (data_path / row.path).resolve())
                for row in labels.itertuples()
            ]
    images_root = data_path / "images"
    if images_root.is_dir():
        return [(path.name, path.resolve()) for path in sorted(images_root.iterdir()) if path.is_dir()]
    return _discover_video_items(data_path)


def _stage2_frame_bounds(path: Path) -> tuple[int, int]:
    if path.is_dir():
        indices = []
        for image_path in path.iterdir():
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            match = re.search(r"(\d+)(?!.*\d)", image_path.stem)
            if match:
                indices.append(int(match.group(1)))
        if not indices:
            raise RuntimeError(f"no numbered frames found in {path}")
        return min(indices), max(indices)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frame_count = int(stream.frames or 0)
        if frame_count <= 0:
            duration = _video_duration(path)
            fps = float(stream.average_rate or stream.base_rate or 0.0)
            frame_count = max(1, round(duration * fps))
    return 0, max(0, frame_count - 1)


def _video_duration(path: Path) -> float:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.duration is not None:
            return float(stream.duration * stream.time_base)
        if container.duration is not None:
            return float(container.duration / av.time_base)
        fps = float(stream.average_rate or stream.base_rate or 0.0)
        return float(stream.frames / fps) if fps > 0 else 0.0


def _video_frame_count(path: Path) -> int:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.frames:
            return int(stream.frames)
        return max(1, sum(1 for _ in container.decode(stream)))


def _resolve_stage_dir(path: Path, stage_name: str) -> Path:
    path = path.expanduser().resolve()
    nested = path / stage_name
    return nested if nested.is_dir() else path


def _find_checkpoint(model_path: Path) -> Path:
    candidates = [model_path / "best.pt", model_path / "stage1" / "best.pt"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Stage 1 checkpoint not found under {model_path}")


def _choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
