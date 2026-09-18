from __future__ import annotations

import json
import sys
from abc import ABC, abstractmethod
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .preprocessing import validate_rgb_frames


class FlowEstimator(ABC):
    @abstractmethod
    def estimate(self, first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return forward flow ``[2,H,W]`` and confidence ``[H,W]``."""

    def estimate_sequence(self, frames: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        validate_rgb_frames(frames)
        height, width = frames[0].shape[:2]
        flows = np.zeros((len(frames), 2, height, width), np.float32)
        confidence = np.ones((len(frames), height, width), np.float32)
        for index in range(1, len(frames)):
            flows[index], confidence[index] = self.estimate(frames[index - 1], frames[index])
        return flows, confidence


class OpenCVFlow(FlowEstimator):
    """Deterministic CPU smoke/ablation backend; never selected implicitly."""

    def estimate(self, first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import cv2

        a = cv2.cvtColor(first, cv2.COLOR_RGB2GRAY)
        b = cv2.cvtColor(second, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 21, 3, 5, 1.2, 0)
        # Photometric agreement after warp is a useful bounded confidence proxy.
        h, w = a.shape
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        warped = cv2.remap(b, (x + flow[..., 0]).astype(np.float32), (y + flow[..., 1]).astype(np.float32), cv2.INTER_LINEAR)
        confidence = np.exp(-np.abs(a.astype(np.float32) - warped) / 24.0)
        return flow.transpose(2, 0, 1).astype(np.float32), confidence.astype(np.float32)


class SeaRaftS(FlowEstimator):
    """Frozen official SEA-RAFT-S with its mixture-Laplace uncertainty head."""

    def __init__(self, source_path: str, checkpoint: str, device: str = "cuda", batch_size: int = 4):
        source = Path(source_path).resolve()
        checkpoint_path = Path(checkpoint).resolve()
        if not (source / "core" / "raft.py").is_file():
            raise FileNotFoundError(f"Official SEA-RAFT source is missing at {source}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"SEA-RAFT checkpoint is missing: {checkpoint_path}")
        core = str(source / "core")
        if core not in sys.path:
            sys.path.insert(0, core)
        with (source / "config" / "eval" / "spring-S.json").open() as stream:
            args = Namespace(**json.load(stream))
        from raft import RAFT

        self.device = torch.device(device)
        self.batch_size = batch_size
        # The official constructor initializes from torchvision before replacing
        # every learned tensor with the SEA-RAFT checkpoint. Disable that network
        # download while retaining the exact architecture.
        import torchvision.models as tv_models

        original_resnet18, original_resnet34 = tv_models.resnet18, tv_models.resnet34
        tv_models.resnet18 = lambda *a, **kw: original_resnet18(weights=None)
        tv_models.resnet34 = lambda *a, **kw: original_resnet34(weights=None)
        try:
            self.model = RAFT(args).to(self.device)
        finally:
            tv_models.resnet18, tv_models.resnet34 = original_resnet18, original_resnet34
        if checkpoint_path.suffix == ".safetensors":
            from safetensors.torch import load_file

            state = load_file(str(checkpoint_path), device="cpu")
        else:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        # ``bn3`` and ``downsample.1`` are two names for the same BatchNorm.
        # safetensors stores only one side of a shared tensor alias.
        allowed_missing = all(
            ".downsample.1." in name and name.split(".downsample.1.")[0] + ".bn3." + name.rsplit(".", 1)[-1] in state
            for name in missing
        )
        if unexpected or (missing and not allowed_missing):
            raise ValueError(f"SEA-RAFT checkpoint mismatch: missing={missing}, unexpected={unexpected}")
        self.model.requires_grad_(False).eval()
        self.iters = int(args.iters)

    @torch.inference_mode()
    def estimate_batch(self, first: torch.Tensor, second: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.model(first, second, iters=self.iters, test_mode=True)
        flow, info = output["flow"][-1], output["info"][-1]
        weights = torch.softmax(info[:, :2].float(), dim=1)
        log_scale = torch.stack(
            (info[:, 2].clamp(0, 10), info[:, 3].clamp(-10, 0)), dim=1
        )
        expected_scale = (weights * log_scale.exp()).sum(1)
        confidence = torch.exp(-expected_scale).clamp(0, 1)
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
        for start in range(1, len(frames), self.batch_size):
            stop = min(len(frames), start + self.batch_size)
            first = torch.stack([torch.from_numpy(frames[i - 1]).permute(2, 0, 1) for i in range(start, stop)]).float().to(self.device)
            second = torch.stack([torch.from_numpy(frames[i]).permute(2, 0, 1) for i in range(start, stop)]).float().to(self.device)
            flow, conf = self.estimate_batch(first, second)
            flows[start:stop] = flow.cpu().numpy()
            confidence[start:stop] = conf.cpu().numpy()
        return flows, confidence


def build_flow_estimator(cfg: dict, device: str | None = None) -> FlowEstimator:
    backend = cfg["backend"]
    if backend == "opencv":
        return OpenCVFlow()
    if backend == "sea_raft":
        return SeaRaftS(
            cfg["source_path"], cfg["checkpoint"], device or cfg.get("device", "cuda"), cfg.get("batch_size", 4)
        )
    raise ValueError(f"Unknown flow backend {backend!r}")


def resize_frames(frames: list[np.ndarray], size: tuple[int, int]) -> list[np.ndarray]:
    import cv2

    height, width = size
    return [cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) for frame in frames]
