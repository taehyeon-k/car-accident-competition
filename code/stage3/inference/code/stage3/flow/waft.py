"""Frozen official WAFT flow backend (Stage 3 v2, Experiment H).

Uses the same mixture-Laplace uncertainty -> confidence mapping as SEA-RAFT so
the downstream geometry/physics code is unchanged. All pretrained sub-modules
(DAv2/timm) are constructed without downloads; the WAFT checkpoint supplies
every tensor and is loaded strictly.
"""
from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from .preprocessing import validate_rgb_frames
from .sea_raft import FlowEstimator


class Waft(FlowEstimator):
    def __init__(self, source_path: str, checkpoint: str, config: str, device: str = "cuda", batch_size: int = 16,
                 iters: int | None = None):
        source = Path(source_path).resolve()
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        with (source / config).open() as stream:
            args = Namespace(**json.load(stream))
        import timm
        import model.backbone.waftv1_dav2 as v1
        import model.backbone.waftv2_dav2 as v2

        original = timm.create_model
        timm.create_model = lambda *a, **kw: original(*a, **{**kw, "pretrained": False})
        inits = {module: module.DepthAnythingFeature.__init__ for module in (v1, v2)}
        for module, init in inits.items():
            module.DepthAnythingFeature.__init__ = (lambda init: lambda self, *a, **kw: init(self, *a, **{**kw, "pretrained": False}))(init)
        try:
            from model import fetch_model
            self.model = fetch_model(args)
        finally:
            timm.create_model = original
            for module, init in inits.items():
                module.DepthAnythingFeature.__init__ = init
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = state.get("model", state) if isinstance(state, dict) else state
        state = {k.removeprefix("module."): v for k, v in state.items()}
        self.model.load_state_dict(state, strict=True)
        self.device = torch.device(device)
        self.model = self.model.to(self.device).requires_grad_(False).eval()
        self.batch_size, self.iters = batch_size, int(iters or args.iters)
        self.var_min, self.var_max = float(getattr(args, "var_min", 0)), float(getattr(args, "var_max", 10))

    @torch.inference_mode()
    def estimate_batch(self, first: torch.Tensor, second: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.model(first, second, iters=self.iters)
        flow, info = output["flow"][-1], output["info"][-1]
        weights = torch.softmax(info[:, :2].float(), dim=1)
        log_scale = torch.stack((info[:, 2].clamp(self.var_min, self.var_max), info[:, 3].clamp(-self.var_max, self.var_min)), dim=1)
        confidence = torch.exp(-(weights * log_scale.exp()).sum(1)).clamp(0, 1)
        return flow.float(), confidence

    def estimate(self, first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tensors = [torch.from_numpy(x).permute(2, 0, 1).float()[None].to(self.device) for x in (first, second)]
        flow, confidence = self.estimate_batch(*tensors)
        return flow[0].cpu().numpy(), confidence[0].cpu().numpy()

    def estimate_sequence(self, frames: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        validate_rgb_frames(frames)
        height, width = frames[0].shape[:2]
        flows = np.zeros((len(frames), 2, height, width), np.float32)
        confidence = np.ones((len(frames), height, width), np.float32)
        stacked = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).to(self.device).float()
        for start in range(1, len(frames), self.batch_size):
            stop = min(len(frames), start + self.batch_size)
            flow, conf = self.estimate_batch(stacked[start - 1:stop - 1], stacked[start:stop])
            flows[start:stop] = flow.cpu().numpy()
            confidence[start:stop] = conf.cpu().numpy()
        return flows, confidence
