from __future__ import annotations

import torch
from torch import nn

from .heads import MotionHeads
from .motion_cnn import MotionCNN
from .physics_mlp import PhysicsMLP
from .tcn import TemporalConvNet
from .v2_modules import (FUSIONS, BiSSM, DualDilatedTCN, GatedTemporalHybrid, MultiScaleMotionEncoder,
                         RefinementStage)


def _temporal(cfg: dict, dim: int) -> nn.Module:
    """v1 ``tcn`` config block plus the optional v2 ``temporal`` selector."""
    tcn = cfg["tcn"]
    kind = cfg.get("temporal", {}).get("type", "tcn")
    extra = cfg.get("temporal", {})
    def plain():
        return TemporalConvNet(dim, tuple(tcn["dilations"]), tcn["dropout"], tcn.get("causal", False),
                               tcn.get("kernel_size", 3), tcn.get("padding_mode", "replicate"))
    def dual():
        pairs = extra.get("dual_pairs", [[1, 2], [2, 4], [4, 8], [8, 16], [16, 32]])
        return DualDilatedTCN(dim, tuple(map(tuple, pairs)), tcn["dropout"], tcn.get("kernel_size", 3))
    def ssm():
        return BiSSM(dim, extra.get("ssm_layers", 4), extra.get("ssm_state", 32), tcn["dropout"])
    if kind == "tcn":
        return plain()
    if kind == "dual_tcn":
        return dual()
    if kind == "ssm":
        return ssm()
    if kind == "tcn_ssm":
        local = dual() if extra.get("local", "dual_tcn") == "dual_tcn" else plain()
        return GatedTemporalHybrid(dim, local, ssm())
    raise ValueError(f"Unknown temporal type {kind!r}")


class Stage3MotionModel(nn.Module):
    """v1 motion TCN. Optional v2 blocks (``motion_encoder``, ``fusion``, ``visual``,
    ``temporal``, ``refinement``, ``auxiliary``) are off unless configured, and
    the v1 parameter names are unchanged so existing checkpoints still load."""

    def __init__(self, cfg: dict):
        super().__init__()
        cnn = cfg["motion_cnn"]
        temporal = cfg["tcn"]
        dim = temporal["dim"]
        encoder = cfg.get("motion_encoder", {})
        self.encoder_type = encoder.get("type", "baseline")
        visual = cfg.get("visual", {})
        self.visual_mode = visual.get("mode") if visual.get("enabled", False) else None
        physics_dim = 32
        self.physics_mlp = PhysicsMLP(20, physics_dim, cfg.get("physics_dropout", 0.05))
        visual_dim = int(visual.get("adapter_dim", 96))
        if self.visual_mode is not None:
            self.visual_adapter = nn.Sequential(nn.LayerNorm(int(visual.get("feature_dim", 384))),
                                                nn.Linear(int(visual.get("feature_dim", 384)), visual_dim), nn.SiLU())
        if self.encoder_type == "baseline":
            self.motion_cnn = MotionCNN(
                10, tuple(cnn["widths"]), cnn.get("blocks_per_stage", 1), cnn.get("pooling", "attention")
            )
            extra = visual_dim if self.visual_mode == "pooled" else 0
            if self.visual_mode == "tokens":
                raise ValueError("visual tokens require motion_encoder.type=multiscale")
            self.fusion = nn.Sequential(nn.Linear(self.motion_cnn.output_dim + physics_dim + extra, dim), nn.LayerNorm(dim))
        elif self.encoder_type == "multiscale":
            token_dim = int(encoder.get("token_dim", 128))
            self.motion_cnn = MultiScaleMotionEncoder(10, tuple(cnn["widths"]), cnn.get("blocks_per_stage", 1),
                                                      token_dim, tuple(encoder.get("grid", [4, 7])))
            if self.visual_mode == "tokens":
                self.visual_to_token = nn.Linear(visual_dim, token_dim)
                self.modality = nn.Parameter(torch.zeros(2, token_dim))
            fusion = cfg.get("fusion", {})
            extra = visual_dim if self.visual_mode == "pooled" else 0
            self.fusion = FUSIONS[fusion.get("type", "concat")](token_dim, physics_dim + extra, dim,
                                                               heads=fusion.get("heads", 4))
        else:
            raise ValueError(f"Unknown motion encoder {self.encoder_type!r}")
        self.temporal = _temporal(cfg, dim)
        auxiliary = cfg.get("auxiliary", {})
        self.heads = MotionHeads(dim, 64, cfg.get("head_dropout", 0.05), cfg.get("yaw_aux", True), auxiliary)
        refinement = cfg.get("refinement", {})
        self.refinement_stages = int(refinement.get("stages", 1)) if refinement.get("enabled", False) else 0
        if self.refinement_stages:
            self.refiners = nn.ModuleList(RefinementStage(dim, refinement.get("blocks", 4), temporal["dropout"])
                                          for _ in range(self.refinement_stages))
            self.refined_heads = nn.ModuleList(MotionHeads(dim, 64, cfg.get("head_dropout", 0.05), cfg.get("yaw_aux", True), auxiliary)
                                               for _ in range(self.refinement_stages))

    def encode_motion(self, motion: torch.Tensor, chunk_frames: int | None = None) -> torch.Tensor:
        if chunk_frames is None:
            return self.motion_cnn(motion)
        if chunk_frames < 1:
            raise ValueError("chunk_frames must be positive")
        device = next(self.parameters()).device
        return torch.cat([self.motion_cnn(motion[:, start:start + chunk_frames].to(device))
                          for start in range(0, motion.shape[1], chunk_frames)], dim=1)

    def forward(self, motion: torch.Tensor, physics: torch.Tensor, lengths: torch.Tensor | None = None,
                chunk_frames: int | None = None, visual: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        spatial = self.encode_motion(motion, chunk_frames)
        physical = self.physics_mlp(physics)
        if self.visual_mode is not None:
            if visual is None:
                raise ValueError("This model requires visual features")
            visual_tokens = self.visual_adapter(visual.to(physical.dtype))
            if self.visual_mode == "pooled":
                physical_extra = torch.cat((physical, visual_tokens.mean(-2)), -1)
        if self.encoder_type == "baseline":
            parts = (spatial, physical) if self.visual_mode is None else (spatial, physical_extra)
            fused = self.fusion(torch.cat(parts, dim=-1))
        else:
            tokens = spatial
            if self.visual_mode == "tokens":
                tokens = torch.cat((tokens + self.modality[0], self.visual_to_token(visual_tokens) + self.modality[1]), -2)
            fused = self.fusion(tokens, physical if self.visual_mode != "pooled" else physical_extra)
        encoded = self.temporal(fused, lengths)
        output = self.heads(encoded)
        if not self.refinement_stages:
            return output
        stages = [output]
        features = encoded
        for refiner, heads in zip(self.refiners, self.refined_heads):
            features = refiner(features, stages[-1], lengths)
            stages.append(heads(features))
        final = dict(stages[-1])
        for level, stage in enumerate(stages[:-1], start=1):
            final.update({f"s{level}_{key}": value for key, value in stage.items()})
        return final
