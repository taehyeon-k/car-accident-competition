"""Offline factories for the workspace checkpoints (no weight downloads)."""

import os
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ASSETS = Path(os.environ.get("STAGE2_PRETRAINED", "/workspace/pretrained"))


def vjepa():
    source = ASSETS / "vjepa2-source"
    if not (source / "app/vjepa_2_1/models/vision_transformer.py").is_file():
        raise FileNotFoundError(
            f"Install the pinned official V-JEPA source at {source}"
        )
    sys.path.insert(0, str(source))
    from app.vjepa_2_1.models.vision_transformer import vit_large

    return vit_large(
        img_size=(384, 384),
        patch_size=16,
        num_frames=64,
        tubelet_size=2,
        use_sdpa=True,
        use_rope=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
        use_activation_checkpointing=True,
    )


def dino():
    from transformers import Dinov2Config, Dinov2Model

    class DenseDINO(Dinov2Model):
        def forward_features(self, x):
            tokens = self(pixel_values=x).last_hidden_state
            return {
                "x_norm_clstoken": tokens[:, 0],
                "x_norm_patchtokens": tokens[:, 1:],
            }

    config = Dinov2Config.from_pretrained(
        ASSETS / "dinov2_base", local_files_only=True
    )
    config._attn_implementation = "sdpa"
    model = DenseDINO(config)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    return model


def depth():
    from transformers import (
        DepthAnythingConfig,
        DepthAnythingForDepthEstimation,
        AutoImageProcessor,
    )

    class NativeDepth(DepthAnythingForDepthEstimation):
        def forward(self, images):
            device = next(self.parameters()).device
            outputs = []
            for image in images:
                inputs = self.processor(images=image.cpu(), return_tensors="pt")
                prediction = super().forward(**inputs.to(device)).predicted_depth
                outputs.append(
                    F.interpolate(
                        prediction[:, None].float(),
                        size=image.shape[-2:],
                        mode="bicubic",
                        align_corners=False,
                    )[0, 0]
                )
            return outputs

    config = DepthAnythingConfig.from_pretrained(
        ASSETS / "depth_anything_v2_small", local_files_only=True
    )
    model = NativeDepth(config)
    model.processor = AutoImageProcessor.from_pretrained(
        ASSETS / "depth_anything_v2_small", local_files_only=True
    )
    return model


def detector():
    from rfdetr.config import RFDETRSmallConfig
    from rfdetr.models.lwdetr import build_model_from_config
    from rfdetr.models import PostProcess

    class NativeDetector(nn.Module):
        def __init__(self):
            super().__init__()
            # A non-null path disables backbone downloads in the low-level builder;
            # build_model_from_config only constructs the model. load_local loads it strictly.
            config = RFDETRSmallConfig(
                pretrain_weights=str(ASSETS / "rfdetr_small/rf-detr-small.pth"),
                num_classes=90,
                device="cpu",
            )
            self.network = build_model_from_config(config)
            self.postprocess = PostProcess(num_select=300)

        def load_state_dict(self, state_dict, strict=True, assign=False):
            # RF-DETR 1.10 adds a derived, nonlearned keypoint mask to older detection checkpoints.
            if "_kp_active_mask" not in state_dict:
                mask = self.network.state_dict()["_kp_active_mask"]
                if mask.numel() != 0:
                    raise ValueError(
                        "Expected a detection-only checkpoint, without keypoints"
                    )
                state_dict = {**state_dict, "_kp_active_mask": mask}
            return self.network.load_state_dict(
                state_dict, strict=strict, assign=assign
            )

        def forward(self, images):
            sizes = torch.tensor(
                [im.shape[-2:] for im in images], device=images[0].device
            )
            mean = images[0].new_tensor([0.485, 0.456, 0.406], dtype=torch.float32)[
                :, None, None
            ]
            std = images[0].new_tensor([0.229, 0.224, 0.225], dtype=torch.float32)[
                :, None, None
            ]
            batch = torch.stack(
                [
                    (
                        F.interpolate(
                            im[None].float() / 255,
                            (512, 512),
                            mode="bilinear",
                            align_corners=False,
                        )[0]
                        - mean
                    )
                    / std
                    for im in images
                ]
            )
            predictions = self.postprocess(self.network(batch), sizes)
            classes = {3: "car", 4: "motorcycle", 6: "bus", 8: "truck"}
            return [
                {
                    "boxes": p["boxes"],
                    "scores": p["scores"],
                    "labels": [classes.get(int(i), "other") for i in p["labels"]],
                }
                for p in predictions
            ]

    return NativeDetector()
