"""LCPyramid (+ optional phase head from phase_study) with complementary auxiliary heads / inputs.

Everything reads the final full-resolution fused hidden sequence h [B,T,H] of the unchanged LCPyramid backbone.
  risk:      r_t = sigmoid(MLP(h_t))                                   (Exp 1)
  boundary:  b_entry(t), b_collision(t) logits                          (Exp 4)
             bnd1: Linear(h_t) | bnd2: Linear([h_t, h_t - h_{t-1}]) | bnd3: Conv1d k=3 over h (local [h_{t-1}, h_t, h_{t+1}])
  lane:      cat4 (OUTSIDE/APPROACHING/CROSSING/INSIDE logits) | ord (1 logit: intrusion score)   (Exp 2)
  motion:    per-position motion input of any dimension, added to the frame embedding like M_motion (Exp 3)
Auxiliary heads are training-only unless an analysis explicitly re-ranks with them; direct ENTRY/COLLISION heads unchanged.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from stage2.phase_study.model import PhasePyramid


class AuxPyramid(PhasePyramid):
    def __init__(self, phase_rep="none", risk=False, boundary="none", lane="none", motion_dim=0, hr=False, clip_norm="none", **kw):
        super().__init__(phase_rep=phase_rep, attach="final", motion=False, **kw)
        H = self.event.in_features
        self.risk = nn.Sequential(nn.Linear(H, 32), nn.GELU(), nn.Linear(32, 1)) if risk else None
        self.boundary_mode = boundary
        if boundary == "bnd1": self.bnd = nn.Linear(H, 2)
        elif boundary == "bnd2": self.bnd = nn.Linear(2 * H, 2)
        elif boundary == "bnd3": self.bnd = nn.Conv1d(H, 2, 3, padding=1)
        else: self.bnd = None
        self.lane_mode = lane
        self.lane = nn.Linear(H, {"cat4": 4, "ord": 1, "none": 0}[lane]) if lane != "none" else None
        self.hr = hr
        # per-clip input normalisation (domain / frame-rate robustness): "x" = centre DINO tokens over the clip's valid positions;
        # "m" = divide motion inputs by their per-clip median magnitude; "xm" = both
        self.clip_norm = clip_norm
        if hr:  # 14x25 high-resolution tokens: per-token LayerNorm + 384->4 projection, flattened (1400) into the input projection
            self.hr_norm = nn.LayerNorm(384); self.hr_tok = nn.Linear(384, 4); motion_dim = motion_dim + 350 * 4
        if motion_dim:
            self.uses_motion = True
            self.motion_proj = nn.Sequential(nn.LayerNorm(motion_dim), nn.Linear(motion_dim, H), nn.GELU(), nn.Dropout(0.35), nn.Linear(H, H))

    def forward(self, x, valid, motion=None, return_hidden=False, hr=None):
        if self.clip_norm != "none":
            vm = valid[..., None, None].float() if x.dim() == 4 else valid[..., None].float()
            if "x" in self.clip_norm:
                x = x.float(); mean = (x * vm).sum(1, keepdim=True) / vm.sum(1, keepdim=True).clamp_min(1)
                x = (x - mean) * vm
            if "m" in self.clip_norm and motion is not None:
                mv = valid[..., None].float(); med = torch.quantile(motion.float().abs().masked_fill(mv == 0, float("nan")).nan_to_num(nan=0.0), 0.5, dim=1, keepdim=True)
                motion = motion.float() / (med + 1e-3)
        if self.hr:
            h_small = self.hr_tok(self.hr_norm(hr.float())).flatten(2)
            motion = h_small if motion is None or motion.shape[-1] == 0 else torch.cat([motion.float(), h_small], -1)
        out = super().forward(x, valid, motion=motion, return_hidden=True)
        h = out["hidden"] * valid[..., None]
        # dropout only when an auxiliary head exists (keeps the RNG stream identical to the plain control)
        hd = self.drop(h) if (self.risk is not None or self.lane is not None or self.boundary_mode in ("bnd1", "bnd3")) else h
        if self.risk is not None: out["risk_logits"] = self.risk(hd).squeeze(-1)
        if self.bnd is not None:
            if self.boundary_mode == "bnd1": b = self.bnd(hd)
            elif self.boundary_mode == "bnd2":
                dh = h - torch.cat([h[:, :1], h[:, :-1]], 1)
                b = self.bnd(self.drop(torch.cat([h, dh], -1)))
            else: b = self.bnd(hd.transpose(1, 2)).transpose(1, 2)
            out["bnd_entry_logits"], out["bnd_collision_logits"] = b[..., 0], b[..., 1]
        if self.lane is not None: out["lane_logits"] = self.lane(hd)
        if not return_hidden: out.pop("hidden")
        return out


def build(cfg):
    return AuxPyramid(phase_rep=cfg.get("phase_rep", "none"), risk=cfg.get("risk", "none") != "none",
                      boundary=cfg.get("boundary", "none"), lane=cfg.get("lane", "none"), motion_dim=cfg.get("motion_dim", 0),
                      hr=cfg.get("hr", False), clip_norm=cfg.get("clip_norm", "none"))


def load(path, device):
    st = torch.load(path, map_location="cpu", weights_only=False)
    m = build(st["config"]); m.load_state_dict(st["model"]); return m.to(device).eval()
