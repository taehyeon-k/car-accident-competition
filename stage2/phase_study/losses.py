"""Composable Stage 2 loss for the phase study.

L = w_direct L_direct + w_phase L_phase + w_tr L_transition + w_mono L_mono + 0.5 L_side + 0.5 L_evasion

L_direct  : NT direct-head loss (Gaussian soft targets, sigma = 1 sampled position) - identical to phase_loss.direct_loss.
L_phase   : cat -> weighted CE (class weights PRE/BETWEEN/POST = 1/3/1) against hard or soft phase targets;
            ord -> weighted BCE on the two cumulative targets y_E(t) = 1[t >= e], y_C(t) = 1[t >= c] (same position weights).
            soft width w > 0 (sampled positions): y_E(t) = Phi((t - e + 0.5) / w); categorical soft target = [1-y_E, y_E-y_C, y_C].
            multiscale: 0.5 * final + 0.5 * mean over coarser levels (targets avg-pooled over each level's windows).
L_transition, L_mono : as in phase_loss.md, computed on the implied 3-class log-probabilities (any representation).
Phase targets use batch["phase_entry_index"/"phase_collision_index"] (== GT, or shuffled for the control arm).
All in sampled-position units; nothing depends on FPS, duration or timestamps.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from stage2.long_context_v2_experiments.phase_loss import attribute_losses, direct_loss, transition_scores
from stage2.spotting_experiments.objective import distribution_loss

PHASE_W = torch.tensor([1.0, 3.0, 1.0])


def cumulative_targets(e, c, length, width):
    t = torch.arange(length, device=e.device, dtype=torch.float32)[None]
    if width <= 0:
        return (t >= e[:, None]).float(), (t >= c[:, None]).float()
    phi = lambda z: 0.5 * (1 + torch.erf(z / math.sqrt(2)))
    return phi((t - e[:, None] + 0.5) / width), phi((t - c[:, None] + 0.5) / width)


def log1mexp(x):  # log(1 - exp(x)) for x < 0
    x = x.clamp_max(-1e-7)
    return torch.where(x > -0.6931, torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))


def phase_nll(raw, rep, yE, yC, weight):
    """Per-position phase loss [B,T] (unreduced, unweighted by validity)."""
    if rep == "cat":
        target = torch.stack([1 - yE, (yE - yC).clamp_min(0), yC], -1)
        return -(target * F.log_softmax(raw.float(), -1)).sum(-1) * weight
    z1, z2 = raw.float().unbind(-1)
    lqE = F.logsigmoid(z1); l1qE = F.logsigmoid(-z1)
    lqC = lqE + F.logsigmoid(z2); l1qC = log1mexp(lqC)
    bce = -(yE * lqE + (1 - yE) * l1qE) - (yC * lqC + (1 - yC) * l1qC)
    return bce * weight


def pooled(x, times):  # avg-pool along T (ceil) `times` times, matching the pyramid's level resolution
    for _ in range(times): x = F.avg_pool1d(x[:, None], 2, stride=2, ceil_mode=True)[:, 0]
    return x


def phase_loss_term(out, batch, rep, width):
    valid = batch["time_valid"].float(); T = valid.shape[1]
    e, c = batch["phase_entry_index"], batch["phase_collision_index"]
    yE, yC = cumulative_targets(e, c, T, width)
    hard = ((torch.arange(T, device=e.device)[None] >= e[:, None]).long() + (torch.arange(T, device=e.device)[None] >= c[:, None]).long())
    w = PHASE_W.to(valid.device)[hard] * valid
    final = (phase_nll(out["phase_raw"], rep, yE, yC, w)).sum() / w.sum()
    if "aux_phase_raw" not in out: return final
    aux = []
    for i, (raw, lv) in enumerate(zip(out["aux_phase_raw"], out["aux_valid"]), 1):
        wv = pooled(w, i); den = pooled(valid, i).clamp_min(1e-6)
        # window-average targets over valid positions only (padding contributes neither mass nor weight)
        yEi, yCi = pooled(yE * valid, i) / den, pooled(yC * valid, i) / den
        wi = wv / den * lv
        aux.append(phase_nll(raw, rep, yEi, yCi, wi).sum() / wi.sum())
    return 0.5 * final + 0.5 * sum(aux) / len(aux)


def transition_mono(out, batch):
    valid, pos = batch["time_valid"], batch["normalized_positions"].float()
    lp = out["phase_logp"]
    # transition_scores applies log_softmax, which is the identity on log-probabilities
    s_e, s_c, lp = transition_scores(lp)
    e, c = batch["phase_entry_index"], batch["phase_collision_index"]
    te = distribution_loss(s_e, e, valid, pos, "soft_index", 1.0); tc = distribution_loss(s_c, c, valid, pos, "soft_index", 1.0)
    l_tr = (0.5 * (te + tc)).mean()
    p = lp.exp(); pair = valid[:, :-1] & valid[:, 1:]
    mono = p[:, :-1, 1] * p[:, 1:, 0] + p[:, :-1, 2] * p[:, 1:, 0] + p[:, :-1, 2] * p[:, 1:, 1]
    return l_tr, (mono * pair).sum() / pair.sum().clamp_min(1)


def total_loss(out, batch, cfg):
    side, evasion = attribute_losses(out, batch)
    parts = {"direct": direct_loss(out, batch)}
    loss = cfg["w_direct"] * parts["direct"] + 0.5 * side + 0.5 * evasion
    if "phase_raw" in out:
        parts["phase"] = phase_loss_term(out, batch, cfg["phase_rep"], cfg["soft_width"])
        parts["transition"], parts["mono"] = transition_mono(out, batch)
        loss = loss + cfg["w_phase"] * parts["phase"] + cfg["w_tr"] * parts["transition"] + cfg["w_mono"] * parts["mono"]
    return loss, parts
