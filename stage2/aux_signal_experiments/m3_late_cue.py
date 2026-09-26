"""Exp 3 / M3: residual motion as a late local event cue only (network unchanged). No training.

For each sampled position t: z_t = robust z-score (per clip, clipped to [0, 6]) of the max of one residual channel over native
frames within +-half the sample spacing (same construction as the camera-shift fusion in decode_motion.shift_scores).
Decode: log p_E + gE * z  ;  log p_C + beta_shift * z_shift + gC * z  ; argmax over e <= c ; native snap as in v5.
Per fold, (beta_shift, channel, gC, gE) are chosen on the OTHER folds (same criterion as decode_motion: NEXAR COLLISION acc +
overall score). The reference uses the identical procedure with gC = gE = 0.
Channels: 0 residual mean, 1 residual p95, 28 centre-road residual, 34 fraction of tracks inconsistent with camera motion.
"""
from __future__ import annotations

import argparse, itertools

import numpy as np

from stage2.long_context_v2_experiments import common as C
from stage2.long_context_v2_experiments.decode_motion import shift_scores
from . import ensemble_test as ET

CHANNELS = {0: "res_mean", 1: "res_p95", 28: "centre_road", 34: "outlier_tracks"}
RES = C.REPO / "stage2/aux_signal_experiments/cache_residual"


def res_z(p, ch, idx, half, zmax=6.0):
    r = np.load(RES / f"{p['sample_id']}.residual.npy", mmap_mode="r")[:, ch]
    s = np.array([r[max(i - half, 0):i + half + 1].max() for i in idx])
    med = np.median(s); mad = np.median(np.abs(s - med)) * 1.4826 + 1e-3
    return np.clip((s - med) / mad, 0, zmax)


def prep(p):
    z_shift, idx, half, fr_all, sh = shift_scores(p)
    p["_z_shift"], p["_idx"], p["_half"], p["_fr_all"], p["_sh"] = z_shift, idx, half, fr_all, sh
    p["_zres"] = {ch: res_z(p, ch, idx, half) for ch in CHANNELS}
    return p


def decode(p, bs, ch, gc, ge):
    fr = np.asarray(p["frames"], float)
    se = p["entry_logp"] + ge * p["_zres"][ch]; sc = p["collision_logp"] + bs * p["_z_shift"] + gc * p["_zres"][ch]
    pair = np.where(np.triu(np.ones((len(fr), len(fr)), bool)), se[:, None] + sc[None], -np.inf)
    i, j = np.unravel_index(np.argmax(pair), pair.shape)
    q = {k: v for k, v in p.items() if not k.startswith("_") and k not in ("entry_logp", "collision_logp", "frames")}
    q["entry_frame"], q["collision_frame"] = int(fr[i]), int(fr[j])
    a0 = p["_idx"][j]; lo = max(a0 - p["_half"], 0); k = lo + int(np.argmax(p["_sh"][lo:a0 + p["_half"] + 1]))
    q["collision_frame"] = int(p["_fr_all"][k]); q["entry_frame"] = min(q["entry_frame"], q["collision_frame"])
    return q


def evaluate(folds, grid):
    cache = {}
    def preds(k, s):
        if (k, s) not in cache: cache[(k, s)] = [decode(p, *s) for p in folds[k]]
        return cache[(k, s)]
    final, chosen = [], []
    for k in folds:
        def other(s):
            ps = [q for j in folds if j != k for q in preds(j, s)]
            return C.metrics([q for q in ps if C.source(q) == "NEXAR"])["collision_acc"] + C.metrics(ps)["score"]
        s = max(grid, key=other); chosen.append(s); final += preds(k, s)
    return C.breakdown(final), chosen


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--family", default=None, help="evaluate a single family draw instead of B3")
    ap.add_argument("--seed-sets", default="0,1"); ap.add_argument("--out", default=None); a = ap.parse_args()
    specs = {"B3": [("C0_avg", None), ("X_ema", None), ("M_motion", None), ("PH_phase", {0, 1})]} if not a.family else \
        {f"{a.family} {d}": [(a.family, set(int(x) for x in d.split(",")))] for d in a.seed_sets.split()}
    betas = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0]
    ref_grid = [(b, 0, 0.0, 0.0) for b in betas]
    cue_grid = [(b, ch, gc, ge) for b, ch, gc, ge in itertools.product(betas, CHANNELS, (0.0, 0.5, 1.0, 2.0), (0.0, 0.5, 1.0)) if gc or ge]
    res = {}
    for name, spec in specs.items():
        folds = {k: [prep(p) for p in ps] for k, ps in ET.pooled(spec).items()}
        b0, c0 = evaluate(folds, ref_grid); b1, c1 = evaluate(folds, ref_grid + cue_grid)
        res[name] = {"reference": b0, "with_residual_cue": b1, "chosen": [list(map(float, x)) for x in c1]}
        for tag, b, ch in (("reference", b0, c0), ("+residual cue", b1, c1)):
            o = b["overall"]
            print(f"{name:18s} {tag:14s} all {o['score']:.4f} E {o['entry_acc']:.3f} C {o['collision_acc']:.3f} NEXAR {b['source:NEXAR']['score']:.4f} "
                  f">1000 {b['bin:>1000']['score']:.4f} | chosen {[(s[0], CHANNELS.get(s[1], '-'), s[2], s[3]) for s in ch]}", flush=True)
    if a.out: C.dump(a.out, res)


if __name__ == "__main__": main()
