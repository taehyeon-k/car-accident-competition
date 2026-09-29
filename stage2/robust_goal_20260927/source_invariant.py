"""Isolated source-adversarial Stage2 head experiment.

The source label is used only by a training loss. Inference inputs and decoded
outputs remain those of the E4_sa model. No shared trainer file is changed.
"""
from __future__ import annotations

import os

import torch
from torch import nn
from torch.nn import functional as F

from stage2.aux_signal_experiments.model import AuxPyramid


SOURCES = ("AIHUB", "CCD", "MMAU", "NEXAR")


class Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.scale * gradient, None


class SourceInvariantPyramid(AuxPyramid):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden = self.event.in_features
        self.source_head = nn.Sequential(nn.Linear(hidden, 64), nn.GELU(), nn.Linear(64, len(SOURCES)))
        self.source_reverse = float(os.environ.get("STAGE2_SOURCE_REVERSE", "0.2"))

    def forward(self, x, valid, motion=None, return_hidden=False, hr=None):
        out = super().forward(x, valid, motion=motion, return_hidden=True, hr=hr)
        h = out["hidden"]
        pooled = (h * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        out["source_logits"] = self.source_head(Reverse.apply(pooled, self.source_reverse))
        if not return_hidden:
            out.pop("hidden", None)
        return out


def load(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    import stage2.aux_signal_experiments.model as model_module
    original = model_module.AuxPyramid
    model_module.AuxPyramid = SourceInvariantPyramid
    try:
        model = model_module.build(checkpoint["config"])
    finally:
        model_module.AuxPyramid = original
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


def main():
    import stage2.aux_signal_experiments.train as trainer
    standard_collate = trainer.collate
    standard_loss = trainer.total_loss

    def collate(items, motion, lane):
        batch = standard_collate(items, motion, lane)
        batch["source_target"] = torch.tensor([
            SOURCES.index(str(item["source_id"]).split(":", 1)[0])
            if str(item["source_id"]).split(":", 1)[0] in SOURCES else -1
            for item in items], dtype=torch.long)
        return batch

    def loss(out, batch, cfg):
        value, parts = standard_loss(out, batch, cfg)
        target = batch["source_target"]
        source_loss = F.cross_entropy(out["source_logits"], target, ignore_index=-1)
        parts["source_ce"] = source_loss
        return value + float(os.environ.get("STAGE2_SOURCE_LOSS", "0.2")) * source_loss, parts

    trainer.AuxPyramid = SourceInvariantPyramid
    trainer.collate = collate
    trainer.total_loss = loss
    trainer.main()


if __name__ == "__main__":
    main()
