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


class CausalEntry(nn.Module):
    """H8: ENTRY logits from frame embeddings using only positions <= t + lookahead (dilated causal convs, residual).
    The bidirectional pyramid (which sees the collision) is not used for ENTRY."""
    def __init__(self, hidden, lookahead=0, dilations=(1, 2, 4, 8, 16), dropout=0.35):
        super().__init__()
        self.lookahead = lookahead
        self.convs = nn.ModuleList(nn.Conv1d(hidden, hidden, 3, dilation=d) for d in dilations)
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in dilations)  # per position: no temporal statistics (GroupNorm leaks the future)
        self.drop = nn.Dropout(dropout); self.out = nn.Linear(hidden, 1)

    def forward(self, emb, valid):  # emb [B,T,H]
        h = (emb * valid[..., None]).transpose(1, 2)
        if self.lookahead: h = F.pad(h, (0, self.lookahead))[..., self.lookahead:]  # input shifted left: output t sees inputs <= t + lookahead
        for conv, norm in zip(self.convs, self.norms):
            d = conv.dilation[0]
            hn = norm(h.transpose(1, 2)).transpose(1, 2)
            h = h + self.drop(F.gelu(conv(F.pad(hn, (2 * d, 0)))))  # left padding only = causal
        return self.out(self.drop(h.transpose(1, 2))).squeeze(-1)


class AuxPyramid(PhasePyramid):
    def __init__(self, phase_rep="none", risk=False, boundary="none", lane="none", motion_dim=0, hr=False, clip_norm="none",
                 causal_entry=-1, anchor_attr=False, entry_aux=False, gap_head=False, entry_bnd="none", entry_bnd_alpha=1.0, **kw):
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
        self.causal_entry = CausalEntry(H, lookahead=causal_entry) if causal_entry >= 0 else None
        # Exp 1: fine-resolution ENTRY boundary branch on level-0 features [l_t, l_t - l_{t-1}] (before pooling), fused as a residual
        self.entry_bnd = entry_bnd; self.entry_bnd_alpha = entry_bnd_alpha
        if entry_bnd != "none":
            self.ebnd_in = nn.Sequential(nn.Linear(2 * H, 64), nn.GELU(), nn.Dropout(0.35)); self.ebnd_conv = nn.Conv1d(64, 1, 3, padding=1)
            if entry_bnd == "gate": self.ebnd_gate = nn.Parameter(torch.tensor(-2.0))  # alpha = alpha_max * sigmoid(g), starts ~0.12
        self.gap_head = nn.Linear(H, 2) if gap_head else None  # gap prior: (mu, log sigma) of log(1 + COLLISION - ENTRY positions)
        self.entry_aux = nn.Linear(H, 1) if entry_aux else None  # P4: broad-target auxiliary ENTRY head (training only, never decoded)
        self.anchor_attr = anchor_attr  # H15: side / evasion pooled at the model's own (detached) ENTRY / COLLISION distributions
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
        if self.causal_entry is not None:  # frame embeddings (same modules as the pyramid input), then the causal ENTRY branch
            emb = self.frame(self.token(self.norm(x.float())).flatten(2))
            if self.uses_motion: emb = emb + self.motion_proj(motion.float())
            neg = torch.finfo(out["entry_logits"].dtype).min / 4
            out["entry_logits"] = self.causal_entry(emb, valid).masked_fill(~valid, neg)
        if self.anchor_attr:
            hid = out["hidden"]; neg = torch.finfo(hid.dtype).min / 4
            wa = self.attn(hid).squeeze(-1).masked_fill(~valid, neg).softmax(-1)
            we = out["entry_logits"].detach().float().masked_fill(~valid, neg).softmax(-1)
            wc = out["collision_logits"].detach().float().masked_fill(~valid, neg).softmax(-1)
            pooled = torch.einsum("bt,bth->bh", (wa + we.to(wa.dtype) + wc.to(wa.dtype)) / 3, hid)
            out["side_logits"] = self.side(self.drop(pooled)); out["evasion_logits"] = self.evasion(self.drop(pooled)).squeeze(-1)
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
        if self.entry_bnd != "none":
            l0 = out["level0"] * valid[..., None]; d = l0 - torch.cat([l0[:, :1], l0[:, :-1]], 1)
            b = self.ebnd_conv(self.ebnd_in(torch.cat([l0, d], -1)).transpose(1, 2)).squeeze(1).float()
            neg = torch.finfo(b.dtype).min / 4; b = b.masked_fill(~valid, neg)
            alpha = self.entry_bnd_alpha * (torch.sigmoid(self.ebnd_gate) if self.entry_bnd == "gate" else 1.0)
            out["entry_v8_logits"] = out["entry_logits"]; out["entry_bnd_logits"] = b
            out["entry_logits"] = (out["entry_logits"].float() + alpha * b.clamp_min(-1e4)).masked_fill(~valid, neg)
        if self.gap_head is not None:  # attention-pooled clip summary (same pooling as side/evasion) -> gap distribution
            hid = out["hidden"]; neg = torch.finfo(hid.dtype).min / 4
            wa = self.attn(hid).squeeze(-1).masked_fill(~valid, neg).softmax(-1)
            g = self.gap_head(self.drop(torch.einsum("bt,bth->bh", wa, hid))).float()
            out["gap_mu"], out["gap_logsig"] = g[:, 0], g[:, 1].clamp(-3, 3)
        if self.entry_aux is not None:
            out["entry_aux_logits"] = self.entry_aux(h).squeeze(-1).masked_fill(~valid, torch.finfo(h.dtype).min / 4)
        if not return_hidden: out.pop("hidden")
        return out


def build(cfg):
    return AuxPyramid(phase_rep=cfg.get("phase_rep", "none"), risk=cfg.get("risk", "none") != "none",
                      boundary=cfg.get("boundary", "none"), lane=cfg.get("lane", "none"), motion_dim=cfg.get("motion_dim", 0),
                      hr=cfg.get("hr", False), clip_norm=cfg.get("clip_norm", "none"), causal_entry=cfg.get("causal_entry", -1),
                      anchor_attr=cfg.get("anchor_attr", False), entry_aux=cfg.get("w_entry_aux", 0.0) > 0, feat_dim=cfg.get("feat_dim", 384), gap_head=cfg.get("w_gap", 0.0) > 0,
                      entry_bnd=cfg.get("entry_bnd", "none"), entry_bnd_alpha=cfg.get("entry_bnd_alpha", 1.0))


def load(path, device):
    st = torch.load(path, map_location="cpu", weights_only=False)
    m = build(st["config"]); m.load_state_dict(st["model"]); return m.to(device).eval()
