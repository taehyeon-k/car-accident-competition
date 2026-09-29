"""Stage2 head with a separate gated pathway for frozen R3D video features.

Only the sampled RGB frames enter the R3D feature extractor. Source, FPS,
timestamps, labels, and duration are not model inputs. Base motion keeps its
own v8-style projection rather than being normalized jointly with R3D.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from stage2.aux_signal_experiments.model import AuxPyramid
from stage2.long_context_v2_experiments import common as C

CACHE = C.REPO / "stage2/robust_goal_20260927/r3d_pilot/cache"
R3D_DIM = 384


class SeparateMotionProjector(nn.Module):
    def __init__(self, base: nn.Module, hidden: int):
        super().__init__()
        self.base = base
        self.r3d = nn.Sequential(nn.LayerNorm(R3D_DIM), nn.Linear(R3D_DIM, hidden), nn.GELU(),
                                 nn.Dropout(0.35), nn.Linear(hidden, hidden))
        self.gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, motion):
        return self.base(motion[..., :-R3D_DIM]) + torch.sigmoid(self.gate) * self.r3d(motion[..., -R3D_DIM:])


class R3DInputPyramid(AuxPyramid):
    def __init__(self, *args, **kwargs):
        kwargs["motion_dim"] -= R3D_DIM
        if kwargs["motion_dim"] != 100:
            raise ValueError(f"Expected the 100-d both-motion baseline, got {kwargs['motion_dim']}")
        super().__init__(*args, **kwargs)
        self.motion_proj = SeparateMotionProjector(self.motion_proj, self.event.in_features)


def append_features(item, stride):
    path = CACHE / f"k{stride}" / f"{item['sample_id']}.npy"
    features = np.load(path, mmap_mode="r")
    if len(features) != len(item["motion"]):
        raise ValueError(f"R3D length mismatch: {path} {len(features)} != {len(item['motion'])}")
    item["motion"] = torch.cat((item["motion"], torch.from_numpy(np.asarray(features, dtype=np.float32))), 1)
    return item


def load(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    import stage2.aux_signal_experiments.model as model_module
    original = model_module.AuxPyramid
    model_module.AuxPyramid = R3DInputPyramid
    try:
        model = model_module.build(checkpoint["config"])
    finally:
        model_module.AuxPyramid = original
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


def main():
    import stage2.aux_signal_experiments.train as trainer
    normal_attach = trainer.attach_inputs
    normal_stride = trainer.stride_item

    def attach(item, motion, lane, geo=False):
        if motion != "both" or lane != "none" or geo:
            raise ValueError("R3D pilot uses the v8 both-motion E4 recipe only")
        return append_features(normal_attach(item, motion, lane, geo), 1)

    def stride(item, k, *args):
        return append_features(normal_stride(item, k, *args), k)

    trainer.attach_inputs = attach
    trainer.stride_item = stride
    trainer.AuxPyramid = R3DInputPyramid
    trainer.main()


if __name__ == "__main__":
    main()
