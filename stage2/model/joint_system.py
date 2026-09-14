"""Online DINOv3 and V-JEPA encoders with LoRA, plus the trainable joint head.

Both backbones keep every pretrained weight frozen and train only the adapters
attached to their last few attention blocks. They consume the *same* augmented
RGB stack produced by the dataset, so neither branch can see a different view of
the clip. Temporal sampling for V-JEPA remains index-only: no FPS is read.
"""

import torch
from torch import nn

from stage2.data.joint_sampling import clip_positions
from stage2.model.backbones import load_local
from stage2.model.joint import JointStage2Model
from stage2.model.joint_tracking import GEOMETRY_DIM
from stage2.model.lora import add_lora_to_last_blocks, transformer_blocks

SCENE_GRID = 4
SCENE_TOKENS = SCENE_GRID * SCENE_GRID + 1


def _attach_lora(encoder, config, prefix):
    """Freeze the backbone, then adapt the last ``*_lora_blocks`` attention sets."""
    encoder.requires_grad_(False)
    blocks = int(config.get(f"{prefix}_lora_blocks", 4))
    if blocks < 1:
        raise ValueError(f"{prefix}_lora_blocks must be positive")
    stack = len(transformer_blocks(encoder))
    attached = add_lora_to_last_blocks(
        encoder,
        rank=int(config.get("lora_rank", 8)),
        alpha=int(config.get("lora_alpha", 16)),
        dropout=float(config.get("lora_dropout", 0.05)),
        blocks=min(blocks, stack),
    )
    if not attached:
        raise ValueError(f"No {prefix} attention projections were adapted")
    return attached


class OnlineDinoLocal(nn.Module):
    """Frozen DINOv3 + LoRA producing scene tokens and per-object ROI features."""

    def __init__(self, encoder, config):
        super().__init__()
        self.encoder = encoder
        self.targets = _attach_lora(self.encoder, config, "dino")
        self.batch_size = int(config.get("dino_batch_size", 8))
        if self.batch_size < 1:
            raise ValueError("dino_batch_size must be positive")

    def forward(self, rgb, roi_boxes, object_valid):
        from torch.nn import functional as F
        from torchvision.ops import roi_align

        b, t = rgb.shape[:2]
        flat = rgb.reshape(b * t, *rgb.shape[2:])
        boxes = roi_boxes.reshape(b * t, *roi_boxes.shape[2:])
        valid = object_valid.reshape(b * t, -1)
        scenes, rois = [], []
        # Chunk the forward pass to bound encoder workspace; gradients still flow.
        for start in range(0, len(flat), self.batch_size):
            images = flat[start : start + self.batch_size]
            output = self.encoder.forward_features(images)
            cls = output["x_norm_clstoken"]
            side = int(round((output["x_norm_patchtokens"].shape[1]) ** 0.5))
            dense = output["x_norm_patchtokens"].reshape(len(images), side, side, -1)
            maps = dense.permute(0, 3, 1, 2)
            cells = F.adaptive_avg_pool2d(maps, (SCENE_GRID, SCENE_GRID))
            scenes.append(
                torch.cat((cls[:, None], cells.flatten(2).transpose(1, 2)), dim=1)
            )
            chunk_valid = valid[start : start + self.batch_size]
            chunk_boxes = boxes[start : start + self.batch_size]
            selected = [
                chunk_boxes[i][chunk_valid[i]].to(maps.dtype)
                for i in range(len(images))
            ]
            pooled = maps.new_zeros(len(images), chunk_valid.shape[1], maps.shape[1])
            if sum(len(x) for x in selected):
                aligned = roi_align(
                    maps,
                    selected,
                    output_size=1,
                    spatial_scale=1.0,
                    aligned=True,
                ).flatten(1)
                cursor = 0
                for i, boxes_i in enumerate(selected):
                    pooled[i, chunk_valid[i]] = aligned[cursor : cursor + len(boxes_i)]
                    cursor += len(boxes_i)
            rois.append(pooled)
        scene = torch.cat(scenes).reshape(b, t, SCENE_TOKENS, -1)
        roi = torch.cat(rois).reshape(b, t, object_valid.shape[-1], -1)
        return scene, roi


class OnlineJointGlobal(nn.Module):
    """Frozen V-JEPA + LoRA over the identical augmented frames."""

    def __init__(self, encoder, config):
        super().__init__()
        self.encoder = encoder
        self.targets = _attach_lora(self.encoder, config, "vjepa")
        self.clip_batch_size = int(config.get("vjepa_clip_batch_size", 1))
        if self.clip_batch_size < 1:
            raise ValueError("V-JEPA clip batch size must be positive")
        if hasattr(self.encoder, "use_activation_checkpointing"):
            self.encoder.use_activation_checkpointing = bool(
                config.get("vjepa_activation_checkpointing", False)
            )

    def forward(self, rgb, time_valid):
        features, times = [], []
        for sample in range(len(rgb)):
            length = int(time_valid[sample].sum())
            if length < 1:
                raise ValueError("Each joint sample needs a valid frame")
            frames = rgb[sample, :length]
            positions = clip_positions(length)
            chunks = []
            for start in range(0, len(positions), self.clip_batch_size):
                group = positions[start : start + self.clip_batch_size]
                video = torch.stack(
                    [
                        frames[clip.to(frames.device)].permute(1, 0, 2, 3)
                        for clip in group
                    ]
                )
                encoded = self.encoder(video)
                if isinstance(encoded, dict):
                    encoded = encoded.get("dense", encoded.get("x"))
                dense = encoded.reshape(len(group), 8, 24, 24, -1)
                chunks.append(dense.mean((2, 3)).reshape(-1, dense.shape[-1]))
            features.append(torch.cat(chunks))
            times.append(
                torch.cat([p.reshape(8, 2).float().mean(1) for p in positions]).to(
                    rgb.device
                )
                / max(1, length - 1)
            )
        maximum = max(len(value) for value in features)
        result = features[0].new_zeros(len(features), maximum, features[0].shape[-1])
        anchors = rgb.new_zeros(len(features), maximum)
        valid = torch.zeros(len(features), maximum, device=rgb.device, dtype=torch.bool)
        for i, (value, time) in enumerate(zip(features, times)):
            result[i, : len(value)] = value
            anchors[i, : len(time)] = time
            valid[i, : len(value)] = True
        return {
            "global_features": result,
            "global_time": anchors,
            "global_valid": valid,
        }


class JointSystem(nn.Module):
    def __init__(self, config, encoder=None, dino=None):
        super().__init__()
        if encoder is None:
            encoder = load_local(
                config["vjepa_factory"],
                config["vjepa_checkpoint"],
                checkpoint_key=config.get("vjepa_checkpoint_key", "ema_encoder"),
            )
        if dino is None:
            dino = load_local(config["dino_factory"], config["dino_checkpoint"])
        self.local_visual = OnlineDinoLocal(dino, config)
        self.global_visual = OnlineJointGlobal(encoder, config)
        self.head = JointStage2Model(
            config.get("geometry_dim", GEOMETRY_DIM),
            config.get("temporal_radius", 16),
            config=config,
        )

    def parameter_groups(self):
        """Split parameters into DINO LoRA, V-JEPA LoRA and joint-head groups."""
        groups = {"dino_lora": [], "vjepa_lora": [], "new": []}
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("local_visual."):
                groups["dino_lora"].append(parameter)
            elif name.startswith("global_visual."):
                groups["vjepa_lora"].append(parameter)
            else:
                groups["new"].append(parameter)
        return groups

    def forward(self, batch):
        scene, roi = self.local_visual(
            batch["rgb"], batch["roi_boxes"], batch["object_valid"]
        )
        features = self.global_visual(batch["rgb"], batch["time_valid"])
        return self.head(
            {**batch, "scene_features": scene, "roi_features": roi, **features}
        )
