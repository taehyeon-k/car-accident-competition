"""Frozen online V-JEPA plus the trainable joint head. Sampling is index-only."""

import torch
from torch import nn
from torchvision.io import ImageReadMode, read_image

from stage2.data.joint_sampling import clip_positions
from stage2.data.transforms import letterbox
from stage2.model.backbones import load_local
from stage2.model.joint import JointStage2Model


class FrozenJointGlobal(nn.Module):
    def __init__(self, encoder, clip_batch_size=1):
        super().__init__()
        if clip_batch_size < 1:
            raise ValueError("V-JEPA clip batch size must be positive")
        self.encoder = encoder.requires_grad_(False).eval()
        self.clip_batch_size = clip_batch_size
        if hasattr(self.encoder, "use_activation_checkpointing"):
            self.encoder.use_activation_checkpointing = False

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, paths_by_sample, device):
        features, times = [], []
        with torch.inference_mode():
            for paths in paths_by_sample:
                positions = clip_positions(len(paths))
                chunks = []
                for start in range(0, len(positions), self.clip_batch_size):
                    group = positions[start : start + self.clip_batch_size]
                    # Decode each used frame once within the clip batch.
                    unique = sorted({int(i) for clip in group for i in clip})
                    frames = {
                        i: letterbox(
                            read_image(str(paths[i]), mode=ImageReadMode.RGB), 384
                        )[0]
                        for i in unique
                    }
                    video = torch.stack(
                        [
                            torch.stack([frames[int(i)] for i in clip], dim=1)
                            for clip in group
                        ]
                    ).to(device)
                    encoded = self.encoder(video)
                    if isinstance(encoded, dict):
                        encoded = encoded.get("dense", encoded.get("x"))
                    dense = encoded.reshape(len(group), 8, 24, 24, 1024)
                    chunks.append(dense.mean((2, 3)).reshape(-1, 1024))
                features.append(torch.cat(chunks))
                times.append(
                    torch.cat([p.reshape(8, 2).float().mean(1) for p in positions]).to(
                        device
                    )
                    / max(1, len(paths) - 1)
                )
        # Allocate outside inference_mode: the trainable projection saves inputs
        # for backward and cannot save inference tensors.
        maximum = max(len(value) for value in features)
        result = torch.zeros(
            len(features), maximum, 1024, device=device, dtype=features[0].dtype
        )
        anchors = torch.zeros(len(features), maximum, device=device)
        valid = torch.zeros(len(features), maximum, device=device, dtype=torch.bool)
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
    def __init__(self, config, encoder=None):
        super().__init__()
        if encoder is None:
            encoder = load_local(
                config["vjepa_factory"],
                config["vjepa_checkpoint"],
                checkpoint_key=config.get("vjepa_checkpoint_key", "ema_encoder"),
            )
        self.global_visual = FrozenJointGlobal(
            encoder, config.get("vjepa_clip_batch_size", 1)
        )
        self.head = JointStage2Model(
            config.get("geometry_dim", 13), config.get("temporal_radius", 16)
        )

    def forward(self, batch):
        features = self.global_visual(
            batch["frame_paths"], batch["scene_features"].device
        )
        return self.head({**batch, **features})
