"""Phase-based temporal loss and decoders (stage2/phase_loss.md).

Phases at sampled positions: PRE t < e*, BETWEEN e* <= t < c*, POST t >= c* (e*, c* = nearest sampled GT indices).
Transition scores (virtual PRE at t = -1 so t = 0 is defined):
  s_E(t) = log p_{t-1}^PRE + log p_t^BETWEEN,   s_C(t) = log p_{t-1}^BETWEEN + log p_t^POST
Losses (all masked to valid positions; Gaussian targets sigma = 1 sampled position via distribution_loss(soft_index)):
  L = 1.0 L_phase(weighted CE 1/3/1) + 0.75 L_transition + 0.25 L_direct + 0.05 L_mono + 0.5 L_side + 0.5 L_evasion
Control ("newtargets"): L = 1.0 L_direct + 0.5 L_side + 0.5 L_evasion (same targets, no phase terms).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from stage2.spotting_experiments.objective import constrained_anchors, distribution_loss

PHASE_W = torch.tensor([1.0, 3.0, 1.0])
LOG_EPS = -13.8  # log(1e-6)


def phase_targets(batch):
    idx = torch.arange(batch["time_valid"].shape[1], device=batch["time_valid"].device)[None]
    e = batch["entry_index"][:, None]; c = batch["collision_index"][:, None]
    return (idx >= e).long() + (idx >= c).long()


def transition_scores(phase_logits):
    lp = F.log_softmax(phase_logits.float(), -1).clamp_min(LOG_EPS)
    pad = torch.tensor([0.0, LOG_EPS, LOG_EPS], device=lp.device).expand(lp.shape[0], 1, 3)
    prev = torch.cat([pad, lp[:, :-1]], 1)
    return prev[..., 0] + lp[..., 1], prev[..., 1] + lp[..., 2], lp


def attribute_losses(out, batch):
    side = F.cross_entropy(out["side_logits"].float(), batch["entry_side"])
    evasion = F.binary_cross_entropy_with_logits(out["evasion_logits"].float(), batch["evasion"].float())
    return side, evasion


def direct_loss(out, batch):
    valid, pos = batch["time_valid"], batch["normalized_positions"].float()
    return 0.5 * (distribution_loss(out["entry_logits"], batch["entry_index"], valid, pos, "soft_index", 1.0).mean()
                  + distribution_loss(out["collision_logits"], batch["collision_index"], valid, pos, "soft_index", 1.0).mean())


def phase_terms(phase_logits, batch):
    valid, pos = batch["time_valid"], batch["normalized_positions"].float()
    y = phase_targets(batch)
    w = PHASE_W.to(phase_logits.device)[y] * valid
    ce = F.cross_entropy(phase_logits.float().transpose(1, 2), y, reduction="none")
    l_phase = (ce * w).sum() / w.sum()
    s_e, s_c, lp = transition_scores(phase_logits)
    ok = (batch["collision_index"] > batch["entry_index"]).float()  # guard: empty BETWEEN (never occurs in this data)
    te = distribution_loss(s_e, batch["entry_index"], valid, pos, "soft_index", 1.0)
    tc = distribution_loss(s_c, batch["collision_index"], valid, pos, "soft_index", 1.0)
    l_tr = (0.5 * (te + tc) * ok).sum() / ok.sum().clamp_min(1)
    p = lp.exp(); pair = valid[:, :-1] & valid[:, 1:]
    mono = p[:, :-1, 1] * p[:, 1:, 0] + p[:, :-1, 2] * p[:, 1:, 0] + p[:, :-1, 2] * p[:, 1:, 1]
    l_mono = (mono * pair).sum() / pair.sum().clamp_min(1)
    return l_phase, l_tr, l_mono


def phase_loss(out, batch, refine_stages=()):
    side, evasion = attribute_losses(out, batch)
    l_direct = direct_loss(out, batch)
    stages = [out["phase_logits"], *refine_stages]  # MS-TCN: supervise every stage equally, average
    parts = [phase_terms(s, batch) for s in stages]
    l_phase = sum(x[0] for x in parts) / len(parts); l_tr = sum(x[1] for x in parts) / len(parts); l_mono = sum(x[2] for x in parts) / len(parts)
    total = 1.0 * l_phase + 0.75 * l_tr + 0.25 * l_direct + 0.05 * l_mono + 0.5 * side + 0.5 * evasion
    return total, {"phase": float(l_phase), "transition": float(l_tr), "direct": float(l_direct), "mono": float(l_mono)}


def newtargets_loss(out, batch):
    side, evasion = attribute_losses(out, batch)
    l_direct = direct_loss(out, batch)
    return 1.0 * l_direct + 0.5 * side + 0.5 * evasion, {"direct": float(l_direct)}


# ------------------------------------------------------------------------------------------------ decoders
def structured_pair_scores(phase_logits, entry_logits=None, collision_logits=None, w_phase=1.0, w_tr=1.0, w_direct=0.0):
    """Score matrix S[e, c] (e < c) for one clip. phase_logits [T,3] (valid positions only)."""
    lp = F.log_softmax(phase_logits.float(), -1).clamp_min(LOG_EPS); t = lp.shape[0]
    zero = lp.new_zeros(1)
    cpre = torch.cat([zero, lp[:, 0].cumsum(0)]); cbtw = torch.cat([zero, lp[:, 1].cumsum(0)]); cpost = torch.cat([zero, lp[:, 2].cumsum(0)])
    e = torch.arange(t, device=lp.device)[:, None]; c = torch.arange(t, device=lp.device)[None]
    s = w_phase * (cpre[e] + (cbtw[c] - cbtw[e]) + (cpost[t] - cpost[c]))
    if w_tr:
        s_e, s_c, _ = transition_scores(phase_logits[None]); s = s + w_tr * (s_e[0][:, None] + s_c[0][None])
    if w_direct and entry_logits is not None:
        s = s + w_direct * (F.log_softmax(entry_logits.float(), -1)[:, None] + F.log_softmax(collision_logits.float(), -1)[None])
    return s.masked_fill(~(e < c), float("-inf"))


def decode_structured(phase_logits, entry_logits=None, collision_logits=None, **w):
    s = structured_pair_scores(phase_logits, entry_logits, collision_logits, **w)
    k = int(torch.argmax(s)); t = s.shape[1]
    return k // t, k % t


def decode_transition(phase_logits):
    s_e, s_c, _ = transition_scores(phase_logits[None])
    e, c = constrained_anchors(s_e, s_c); return int(e[0]), int(c[0])


def decode_direct(entry_logits, collision_logits):
    e, c = constrained_anchors(entry_logits[None].float(), collision_logits[None].float()); return int(e[0]), int(c[0])
