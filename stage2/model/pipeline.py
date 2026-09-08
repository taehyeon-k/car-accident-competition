"""End-to-end learned branches: visual LoRA remains live during training."""

from __future__ import annotations
import torch
import torch.nn as nn
from .backbones import VJEPAAdapter, DINOAdapter
from .model import CoarseModel, FineModel


class CoarseSystem(nn.Module):
    """Keep V-JEPA LoRA in the training graph, then run the coarse head."""

    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.visual = VJEPAAdapter(
            config["vjepa_factory"],
            config["vjepa_checkpoint"],
            config["lora_rank"],
            config["lora_alpha"],
            config["lora_dropout"],
            checkpoint_key=config.get(
                "vjepa_checkpoint_key",
                "ema_encoder",
            ),
        )
        self.head = CoarseModel()

    def forward(
        self,
        batch,
    ):
        return self.head(
            self.visual(batch["coarse_rgb"]),
            batch["boxes_grid"],
            batch["geometry"],
            batch["object_valid"].bool(),
            batch["bin_valid"].bool(),
        )


class FineSystem(nn.Module):
    """Encode valid native frames in chunks, preserving LoRA gradients."""

    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.visual = DINOAdapter(
            config["dino_factory"],
            config["dino_checkpoint"],
            config["lora_rank"],
            config["lora_alpha"],
            config["lora_dropout"],
        )
        self.head = FineModel()
        self.frame_batch_size = config.get(
            "frame_batch_size",
            8,
        )
        if self.frame_batch_size < 1:
            raise ValueError("frame_batch_size must be positive")

    def forward(
        self,
        batch,
    ):
        batch_size, window_length = batch["fine_rgb"].shape[:2]
        rgb = batch["fine_rgb"].flatten(
            0,
            1,
        )
        valid = batch["time_valid"].bool().flatten()
        indices = valid.nonzero(as_tuple=True)[0]
        if indices.numel() == 0:
            raise ValueError("Fine input contains no real native frames")

        # Padded images never enter DINO. Chunking bounds per-forward attention
        # workspaces; autograd still retains training activations for real frames.
        global_chunks = []
        dense_chunks = []
        for chunk_indices in indices.split(self.frame_batch_size):
            global_features, patch_features = self.visual(rgb[chunk_indices])
            global_chunks.append(global_features)
            dense_chunks.append(patch_features)

        global_features = torch.cat(global_chunks)
        patch_features = torch.cat(dense_chunks)
        global_token = global_features.new_zeros(
            batch_size * window_length,
            384,
        )
        dense = patch_features.new_zeros(
            batch_size * window_length,
            24,
            24,
            384,
        )
        global_token = global_token.index_copy(
            0,
            indices,
            global_features,
        )
        dense = dense.index_copy(
            0,
            indices,
            patch_features,
        )
        return self.head(
            global_token.reshape(
                batch_size,
                window_length,
                384,
            ),
            dense.reshape(
                batch_size,
                window_length,
                24,
                24,
                384,
            ),
            batch["boxes_grid"],
            batch["geometry"],
            batch["object_valid"].bool(),
            batch["time_valid"].bool(),
        )
