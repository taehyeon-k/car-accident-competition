"""Isolated Stage2 experiment: scene-conditioned actor selection for ENTRY.

Unlike the existing OB_D branch, this branch uses the scene state as a query
over tracked actors at each position. It changes ENTRY only; collision and
attributes retain the base motion pathway. The common trainer is reused via
an in-process model-class substitution, leaving shared training code untouched.
"""
from __future__ import annotations

import torch
from torch import nn

from stage2.aux_signal_experiments.model import AuxPyramid


class ActorInteractionPyramid(AuxPyramid):
    def __init__(self, *args, **kwargs):
        branch = kwargs.pop("obj_branch", "gate")
        self_k = kwargs.pop("obj_k", 3)
        self_d = kwargs.pop("obj_d", 48)
        self_alpha = kwargs.pop("obj_alpha", 1.0)
        kwargs.pop("obj_state", None)
        kwargs.pop("obj_lane_drop", None)
        kwargs.pop("obj_rgate", None)
        if branch != "gate" or self_d <= 0:
            raise ValueError("Actor interaction requires --obj-branch gate and --obj-d > 0")
        kwargs["motion_dim"] -= 2 * self_k * self_d
        super().__init__(*args, obj_branch="none", **kwargs)
        hidden = self.event.in_features
        self.actor_k, self.actor_d, self.actor_alpha = self_k, self_d, self_alpha
        self.actor_in = nn.Sequential(nn.LayerNorm(2 * self_d), nn.Linear(2 * self_d, 64), nn.GELU())
        self.actor_query = nn.Linear(hidden, 64)
        self.actor_key = nn.Linear(64, 64)
        self.actor_value = nn.Linear(64, 64)
        self.actor_temporal = nn.Sequential(
            nn.Conv1d(128, 64, 5, padding=2), nn.GELU(), nn.Dropout(0.35),
            nn.Conv1d(64, 1, 3, padding=1),
        )
        self.actor_gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x, valid, motion=None, return_hidden=False, hr=None):
        if motion is None:
            raise ValueError("Actor features must be appended to motion")
        width = 2 * self.actor_k * self.actor_d
        obj, base = motion[..., -width:].float(), motion[..., :-width]
        out = super().forward(x, valid, motion=base, return_hidden=True, hr=hr)
        b, t, _ = obj.shape
        k, d = self.actor_k, self.actor_d
        mx = obj[..., :k * d].reshape(b, t, k, d)
        mean = obj[..., k * d:].reshape(b, t, k, d)
        present = mx[..., 0] > 0
        actors = self.actor_in(torch.cat((mx, mean), -1))
        query = self.actor_query(out["hidden"]).unsqueeze(2)
        key = self.actor_key(actors)
        score = (query * key).sum(-1) / 8.0
        weights = score.masked_fill(~present, -1e4).softmax(-1)
        weights = weights * present.any(-1, keepdim=True)
        context = (weights[..., None] * self.actor_value(actors)).sum(2)
        delta = context - torch.cat((context[:, :1], context[:, :-1]), 1)
        branch = self.actor_temporal(torch.cat((context, delta), -1).transpose(1, 2)).squeeze(1)
        branch = branch.masked_fill(~present.any(-1), 0.0)
        neg = torch.finfo(torch.float32).min / 4
        out["entry_obj_logits"] = branch.masked_fill(~valid, neg)
        out["entry_v8_logits"] = out["entry_logits"]
        out["entry_logits"] = (out["entry_logits"].float() + self.actor_alpha * torch.sigmoid(self.actor_gate) * branch).masked_fill(~valid, neg)
        if not return_hidden:
            out.pop("hidden", None)
        return out


def load(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    from stage2.aux_signal_experiments.model import build
    cfg = checkpoint["config"]
    # Reuse the canonical model constructor's arguments, with this class selected.
    import stage2.aux_signal_experiments.model as model_module
    original = model_module.AuxPyramid
    model_module.AuxPyramid = ActorInteractionPyramid
    try:
        model = build(cfg)
    finally:
        model_module.AuxPyramid = original
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


def main():
    import stage2.aux_signal_experiments.train as trainer
    trainer.AuxPyramid = ActorInteractionPyramid
    trainer.main()


if __name__ == "__main__":
    main()
