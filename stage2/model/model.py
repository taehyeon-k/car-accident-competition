"""Learned coarse and fine localization heads.

The backbones live in ``backbones.py``. These modules only consume dense visual
features, so tensor shapes and masking can be inspected without loading weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.ops import roi_align

from .geometry import GeometryMLP
from .modules import ResidualTemporalConv, Transformer, VideoPool, sinusoidal


def roi_appearance(
    dense_features: torch.Tensor,
    boxes_in_grid: torch.Tensor,
    object_valid: torch.Tensor,
) -> torch.Tensor:
    """Apply aligned 3x3 ROIAlign and mean pool valid persistent tracks.

    Args:
        dense_features: ``[B, T, H, W, C]`` V-JEPA or DINO patch features.
        boxes_in_grid: ``[B, T, N, 4]`` boxes in the patch-grid coordinate system.
        object_valid: ``[B, T, N]`` persistent-track visibility mask.

    Returns:
        ``[B, T, N, C]`` appearance features. Invalid positions are exactly zero.
    """
    batch, time, grid_h, grid_w, channels = dense_features.shape
    num_objects = boxes_in_grid.shape[2]
    output = dense_features.new_zeros(
        batch,
        time,
        num_objects,
        channels,
    )

    valid_indices = object_valid.nonzero(as_tuple=False)
    if len(valid_indices) == 0:
        return output

    # ROIAlign expects [image, channel, height, width]. One image is one time step.
    feature_map = dense_features.permute(
        0,
        1,
        4,
        2,
        3,
    )
    feature_map = feature_map.reshape(
        batch * time,
        channels,
        grid_h,
        grid_w,
    )

    image_index = (valid_indices[:, 0] * time + valid_indices[:, 1]).float()
    selected_boxes = boxes_in_grid[
        valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]
    ]
    rois = torch.cat(
        (image_index[:, None], selected_boxes),
        dim=1,
    )

    # torchvision's native ROIAlign does not support BF16 on every backend.
    # Restrict the fallback to this operator; gradients still reach the backbone.
    with torch.autocast(
        device_type=feature_map.device.type,
        enabled=False,
    ):
        pooled = roi_align(
            feature_map.float(),
            rois.float(),
            output_size=(3, 3),
            spatial_scale=1.0,
            sampling_ratio=2,
            aligned=True,
        ).mean(dim=(-1, -2))
    pooled = pooled.to(output.dtype)

    output[valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]] = pooled
    return output


class CoarseModel(nn.Module):
    """32-bin event model operating on 16 V-JEPA tubelet feature maps."""

    def __init__(self) -> None:
        super().__init__()

        # Global scene and object appearance projections.
        self.global_projection = nn.Sequential(
            nn.Linear(
                768,
                384,
            ),
            nn.LayerNorm(384),
        )
        self.roi_projection = nn.Sequential(
            nn.Linear(
                768,
                256,
            ),
            nn.LayerNorm(256),
        )
        self.geometry_embedding = GeometryMLP()

        # Interaction first happens inside each tubelet, then across tubelets.
        self.spatial_transformer = Transformer(num_layers=2)
        self.temporal_transformer = Transformer(num_layers=4)

        # Each tubelet emits two logits: its first and second representative frame.
        self.entry_head = nn.Sequential(
            nn.Linear(
                384,
                128,
            ),
            nn.GELU(),
            nn.Linear(
                128,
                2,
            ),
        )
        self.collision_head = nn.Sequential(
            nn.Linear(
                384,
                128,
            ),
            nn.GELU(),
            nn.Linear(
                128,
                2,
            ),
        )

        self.video_pool = VideoPool()
        self.direction_head = nn.Sequential(
            nn.Linear(
                384,
                128,
            ),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(
                128,
                2,
            ),
        )
        self.evasion_head = nn.Sequential(
            nn.Linear(
                384,
                128,
            ),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(
                128,
                1,
            ),
        )

    def forward(
        self,
        dense_features: torch.Tensor,
        boxes_in_grid: torch.Tensor,
        tubelet_geometry: torch.Tensor,
        object_valid: torch.Tensor,
        bin_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return coarse entry/collision/direction/evasion outputs."""
        batch, tubelets, _, _, _ = dense_features.shape
        expected_geometry_shape = (12, 9)
        if (
            tuple(dense_features.shape[1:]) != (16, 24, 24, 768)
            or tuple(tubelet_geometry.shape[2:]) != expected_geometry_shape
        ):
            raise ValueError("Expected dense [B,16,24,24,768] and geometry [B,16,12,9]")

        if not bin_valid.any(dim=1).all():
            raise ValueError("Every coarse sample must have a valid bin")
        tubelet_valid = bin_valid.reshape(
            batch,
            16,
            2,
        ).any(dim=-1)
        object_valid = object_valid & tubelet_valid[..., None]
        tubelet_geometry = tubelet_geometry.masked_fill(
            ~object_valid[..., None],
            0,
        )

        # Global token is a spatial mean of each dense V-JEPA feature grid.
        global_token = self.global_projection(dense_features.mean(dim=(2, 3)))

        roi_features = roi_appearance(
            dense_features,
            boxes_in_grid,
            object_valid,
        )
        roi_token = self.roi_projection(roi_features)
        geometry_token = self.geometry_embedding(tubelet_geometry)
        object_tokens = torch.cat(
            (roi_token, geometry_token),
            dim=-1,
        )
        object_tokens = object_tokens * object_valid[..., None]

        # Token zero is GLOBAL. Masked object slots cannot become attention keys or values.
        spatial_tokens = torch.cat(
            (global_token[:, :, None], object_tokens),
            dim=2,
        )
        spatial_tokens = spatial_tokens.reshape(
            batch * tubelets,
            13,
            384,
        )
        global_valid = torch.ones(
            batch,
            tubelets,
            1,
            dtype=torch.bool,
            device=dense_features.device,
        )
        spatial_valid = torch.cat(
            (global_valid, object_valid),
            dim=2,
        )
        spatial_valid = spatial_valid.reshape(
            batch * tubelets,
            13,
        )

        interaction = self.spatial_transformer(
            spatial_tokens,
            spatial_valid,
        )[:, 0]
        interaction = interaction.reshape(
            batch,
            tubelets,
            384,
        )

        tubelet_valid = bin_valid.reshape(
            batch,
            16,
            2,
        ).any(dim=-1)
        temporal_hidden = self.temporal_transformer(
            interaction
            + sinusoidal(
                tubelet_valid,
                coarse=True,
            ).to(interaction.dtype),
            tubelet_valid,
        )

        entry_logits = self.entry_head(temporal_hidden).reshape(
            batch,
            32,
        )
        collision_logits = self.collision_head(temporal_hidden).reshape(
            batch,
            32,
        )
        entry_logits = entry_logits.masked_fill(
            ~bin_valid,
            -torch.inf,
        )
        collision_logits = collision_logits.masked_fill(
            ~bin_valid,
            -torch.inf,
        )

        video_embedding = self.video_pool(
            temporal_hidden,
            tubelet_valid,
        )
        return {
            "entry_logits": entry_logits,
            "collision_logits": collision_logits,
            "direction_logits": self.direction_head(video_embedding),
            "evasion_logits": self.evasion_head(video_embedding),
            "hidden": temporal_hidden,
        }


class FineModel(nn.Module):
    """Exact-frame localizer operating on a padded 64-frame native window."""

    def __init__(self) -> None:
        super().__init__()

        self.roi_projection = nn.Sequential(
            nn.Linear(
                384,
                256,
            ),
            nn.LayerNorm(256),
        )
        self.geometry_embedding = GeometryMLP()
        self.spatial_transformer = Transformer(num_layers=2)

        self.conv_kernel_3 = ResidualTemporalConv(kernel_size=3)
        self.conv_kernel_5 = ResidualTemporalConv(kernel_size=5)
        self.temporal_transformer = Transformer(num_layers=2)

        self.boundary_projection = nn.Linear(
            1152,
            384,
        )
        self.boundary_norm = nn.LayerNorm(384)

        self.entry_head = nn.Sequential(
            nn.Linear(
                384,
                128,
            ),
            nn.GELU(),
            nn.Linear(
                128,
                1,
            ),
        )
        self.collision_head = nn.Sequential(
            nn.Linear(
                384,
                128,
            ),
            nn.GELU(),
            nn.Linear(
                128,
                1,
            ),
        )
        self.entry_state_head = nn.Linear(
            384,
            1,
        )
        self.collision_state_head = nn.Linear(
            384,
            1,
        )

    def forward(
        self,
        global_tokens: torch.Tensor,
        dense_features: torch.Tensor,
        boxes_in_grid: torch.Tensor,
        geometry: torch.Tensor,
        object_valid: torch.Tensor,
        time_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return exact-frame and auxiliary before/after-state logits."""
        batch, frames, _ = global_tokens.shape

        if not time_valid.any(dim=1).all():
            raise ValueError("Every fine window must contain a valid native frame")
        object_valid = object_valid & time_valid[..., None]
        geometry = geometry.masked_fill(
            ~object_valid[..., None],
            0,
        )
        global_tokens = global_tokens.masked_fill(
            ~time_valid[..., None],
            0,
        )

        roi_features = roi_appearance(
            dense_features,
            boxes_in_grid,
            object_valid,
        )
        object_tokens = torch.cat(
            (self.roi_projection(roi_features), self.geometry_embedding(geometry)),
            dim=-1,
        )
        object_tokens = object_tokens * object_valid[..., None]

        spatial_tokens = torch.cat(
            (global_tokens[:, :, None], object_tokens),
            dim=2,
        )
        spatial_tokens = spatial_tokens.reshape(
            batch * frames,
            13,
            384,
        )
        spatial_valid = torch.cat(
            (time_valid[:, :, None], object_valid),
            dim=2,
        )
        spatial_valid = spatial_valid.reshape(
            batch * frames,
            13,
        )

        frame_states = self.spatial_transformer(
            spatial_tokens,
            spatial_valid,
        )[:, 0]
        frame_states = frame_states.reshape(
            batch,
            frames,
            384,
        )
        frame_states = frame_states.masked_fill(
            ~time_valid[..., None],
            0,
        )

        temporal_states = self.conv_kernel_3(
            frame_states,
            time_valid,
        )
        temporal_states = self.conv_kernel_5(
            temporal_states,
            time_valid,
        )
        temporal_states = self.temporal_transformer(
            temporal_states + sinusoidal(time_valid).to(temporal_states.dtype),
            time_valid,
        )

        # The first frame uses itself as the previous state by construction.
        previous_frame = torch.cat(
            (frame_states[:, :1], frame_states[:, :-1]),
            dim=1,
        )
        boundary_input = torch.cat(
            (previous_frame, frame_states, frame_states - previous_frame),
            dim=-1,
        )
        final_states = self.boundary_norm(
            temporal_states + self.boundary_projection(boundary_input)
        )

        padded = ~time_valid
        return {
            "entry_logits": self.entry_head(final_states)
            .squeeze(-1)
            .masked_fill(
                padded,
                -torch.inf,
            ),
            "collision_logits": self.collision_head(final_states)
            .squeeze(-1)
            .masked_fill(
                padded,
                -torch.inf,
            ),
            "entry_state_logits": self.entry_state_head(final_states).squeeze(-1),
            "collision_state_logits": self.collision_state_head(final_states).squeeze(
                -1
            ),
        }
