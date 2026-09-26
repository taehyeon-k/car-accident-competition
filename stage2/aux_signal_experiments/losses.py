"""Auxiliary losses for the complementary-signal study. All targets use sampled-position indices only (no FPS/seconds).

Base loss: 'nt' = NT recipe (1.0 direct Gaussian sigma=1 + 0.5 side + 0.5 evasion, = phase_study A0) optionally with the
phase_study phase terms; 'p2' = the P2 / M_motion loss (temporal_pyramid_experiments.experiment_loss).

Risk (Exp 1), r_t = sigmoid(logit):
  R1  BCE to y_t = 0 (t < e), (t - e) / (c - e) (e <= t < c), 1 (t >= c)
  R2  R1 + future consistency: mean relu(sg(r_t) - r_{t+d}), d in {1,2,3}, pairs with t + d <= c  (later >= earlier, earlier frozen)
  R3  R1 + adaptive monotonic: mean relu(r_t - r_{t+1} - 0.1) for t + 1 <= c  (only large drops before COLLISION)
Boundary (Exp 4): balanced BCE per position against Gaussian peaks exp(-0.5 ((t - e) / w)^2) (w in sampled positions),
  positive term weighted by n_valid / sum(target) (capped at 50), averaged over ENTRY and COLLISION heads.
Lane (Exp 2): cat4 -> CE to per-position pseudo-label state (ignore = -1); ord -> BCE to the pseudo-label intrusion score in [0,1];
  transition (L3) -> soft-argmax transition score on cat4 probabilities, supervised at the pseudo-label CROSSING onset.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from stage2.phase_study.losses import total_loss as nt_total
from stage2.temporal_pyramid_experiments.train import experiment_loss

P2_CFG = dict(event_target="soft_normalized", target_width=.015, target_width_mode="normalized_fixed",
              hard_negative_weight=0.0, recipe="P2")


def risk_targets(e, c, T):
    t = torch.arange(T, device=e.device, dtype=torch.float32)[None]
    e = e[:, None].float(); c = c[:, None].float()
    return ((t - e) / (c - e).clamp_min(1)).clamp(0, 1)


def risk_loss(out, batch, mode):
    valid = batch["time_valid"].float(); T = valid.shape[1]
    z = out["risk_logits"].float(); y = risk_targets(batch["entry_index"], batch["collision_index"], T)
    loss = (F.binary_cross_entropy_with_logits(z, y, reduction="none") * valid).sum() / valid.sum()
    r = torch.sigmoid(z); t = torch.arange(T, device=z.device)[None]; c = batch["collision_index"][:, None]
    if mode == "R2":
        terms = []
        for d in (1, 2, 3):
            ok = (valid[:, :-d] * valid[:, d:] * (t[:, d:] <= c).float())
            terms.append((F.relu(r[:, :-d].detach() - r[:, d:]) * ok).sum() / ok.sum().clamp_min(1))
        loss = loss + sum(terms) / len(terms)
    elif mode == "R3":
        ok = valid[:, :-1] * valid[:, 1:] * (t[:, 1:] <= c).float()
        loss = loss + (F.relu(r[:, :-1] - r[:, 1:] - 0.1) * ok).sum() / ok.sum().clamp_min(1)
    return loss


def boundary_loss(out, batch, width):
    valid = batch["time_valid"].float(); T = valid.shape[1]
    t = torch.arange(T, device=valid.device, dtype=torch.float32)[None]
    total = 0.0
    for ev in ("entry", "collision"):
        g = torch.exp(-0.5 * ((t - batch[f"{ev}_index"][:, None].float()) / width) ** 2) * valid
        pos_w = (valid.sum(1, keepdim=True) / g.sum(1, keepdim=True).clamp_min(1e-6)).clamp(max=50)
        z = out[f"bnd_{ev}_logits"].float()
        l = -(pos_w * g * F.logsigmoid(z) + (1 - g) * F.logsigmoid(-z))
        total = total + (l * valid).sum() / valid.sum()
    return total / 2


def lane_loss(out, batch, mode, transition_weight=0.0):
    valid = batch["time_valid"]; z = out["lane_logits"].float()
    if mode == "cat4":
        y = batch["lane_state"].clone(); y[~valid] = -1
        loss = F.cross_entropy(z.transpose(1, 2), y, ignore_index=-1)
        if transition_weight:
            lp = F.log_softmax(z, -1)
            before = torch.logsumexp(lp[..., :2], -1); after = torch.logsumexp(lp[..., 2:], -1)
            prev = torch.cat([torch.zeros_like(before[:, :1]), before[:, :-1]], 1)
            s = (prev + after).masked_fill(~valid, -1e4)
            onset = batch["lane_onset"]; ok = onset >= 0
            if ok.any():
                loss = loss + transition_weight * F.cross_entropy(s[ok], onset[ok])
        return loss
    y = batch["lane_score"]; m = valid.float() * (y >= 0).float()
    return (F.binary_cross_entropy_with_logits(z[..., 0], y.clamp(0, 1), reduction="none") * m).sum() / m.sum().clamp_min(1)


def total_loss(out, batch, cfg):
    if cfg.get("base_loss", "nt") == "p2":
        loss, parts = experiment_loss(out, batch, {**cfg, **P2_CFG}); parts = {"base": float(loss)}
    else:
        loss, parts = nt_total(out, batch, cfg)
    if cfg.get("risk", "none") != "none":
        parts["risk"] = risk_loss(out, batch, cfg["risk"]); loss = loss + cfg["w_risk"] * parts["risk"]
    if cfg.get("boundary", "none") != "none":
        parts["boundary"] = boundary_loss(out, batch, cfg["bnd_width"]); loss = loss + cfg["w_bnd"] * parts["boundary"]
    if cfg.get("lane", "none") != "none":
        parts["lane"] = lane_loss(out, batch, cfg["lane"], cfg.get("w_lane_tr", 0.0)); loss = loss + cfg["w_lane"] * parts["lane"]
    return loss, parts
