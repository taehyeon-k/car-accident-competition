"""Late fusion of the raw camera-shift signal into COLLISION decoding (saved OOF CV logits; no training).

Per sampled position t: s_t = max global-shift magnitude over native frames within half a sample spacing of t.
Per clip robust z-score z_t = (s_t - median) / (1.4826*MAD + eps), clipped to [0, zmax].
collision score += beta * z_t   (optionally only when N > 500 frames).  Optional pos+gap priors from decode_priors.
beta (and gating) chosen per fold on the OTHER folds' OOF clips (never on the evaluated fold).
Optional native snap: move the chosen collision frame to the shift argmax within +-half spacing.
Frame numbers / counts only; no FPS.
"""
from __future__ import annotations

import argparse, itertools, json

import numpy as np

from . import common as C
from .decode_priors import LONG, fit, lognorm, oof

R = C.REPO / "stage2/long_context_v2_experiments/results"


def shift_scores(p, zmax=6.0):
    fr_all = np.load(C.DENSE / f"{p['sample_id']}.frames.npy"); sh = C.motion(p["sample_id"])[:, 2]
    fr = np.asarray(p["frames"]); idx = np.searchsorted(fr_all, fr)
    half = max(1, int(np.ceil(np.median(np.diff(idx)) / 2))) if len(idx) > 1 else 1
    s = np.array([sh[max(i - half, 0):i + half + 1].max() for i in idx])
    med = np.median(s); mad = np.median(np.abs(s - med)) * 1.4826 + 1e-3
    return np.clip((s - med) / mad, 0, zmax), idx, half, fr_all, sh


def decode(p, prior, a, g, beta, gate_long, snap):
    fr = np.asarray(p["frames"], float); pos = (fr - fr[0]) / max(fr[-1] - fr[0], 1)
    se, sc = p["entry_logp"].copy(), p["collision_logp"].copy()
    long = p["num_available_frames"] > LONG
    z, idx, half, fr_all, sh = shift_scores(p)
    if beta and (long or not gate_long): sc = sc + beta * z
    g_term = 0
    if prior is not None and long:
        se = se + a * lognorm(pos, prior["e"]); sc = sc + a * lognorm(pos, prior["c"])
        if g: g_term = g * lognorm(pos[None, :] - pos[:, None], prior["g"])
    pair = se[:, None] + sc[None, :] + g_term
    pair = np.where(np.triu(np.ones_like(pair, bool)), pair, -np.inf)
    i, j = np.unravel_index(np.argmax(pair), pair.shape)
    q = {k: v for k, v in p.items() if k not in ("entry_logits", "collision_logits", "entry_logp", "collision_logp", "frames")}
    q["entry_frame"], q["collision_frame"] = int(fr[i]), int(fr[j])
    if snap and (long or not gate_long):
        a0 = idx[j]; lo = max(a0 - half, 0); k = lo + int(np.argmax(sh[lo:a0 + half + 1]))
        q["collision_frame"] = int(fr_all[k]); q["entry_frame"] = min(q["entry_frame"], q["collision_frame"])
    return q


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); a_ = ap.parse_args()
    betas = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0]
    out = {}
    for run in a_.runs:
        folds = oof(run)
        priors = {k: fit(C.rows(str(C.REPO / f"stage2/long_context_v2_experiments/folds/fold{k}_train.jsonl"))) for k in folds}
        print(f"\n== {run}")
        out[run] = {}
        for use_prior, gate_long, snap in itertools.product((False, True), (True, False), (False, True)):
            cache = {}
            def preds_for(k, beta):
                key = (k, beta)
                if key not in cache:
                    pr = priors[k] if use_prior else None
                    cache[key] = [decode(p, pr, 0.5, 1.0 if use_prior else 0, beta, gate_long, snap) for p in folds[k]]
                return cache[key]
            final, chosen = [], {}
            for k in folds:  # choose beta on other folds only
                def other_score(beta):
                    ps = [q for j in folds if j != k for q in preds_for(j, beta)]
                    return C.metrics([q for q in ps if C.source(q) == "NEXAR"])["collision_acc"] + C.metrics(ps)["score"]
                b = max(betas, key=other_score); chosen[k] = b; final += preds_for(k, b)
            bd = C.breakdown(final); n = bd["source:NEXAR"]
            name = f"prior={int(use_prior)} gate_long={int(gate_long)} snap={int(snap)}"
            out[run][name] = {"breakdown": bd, "beta_per_fold": chosen}
            print(f"  {name:32s} beta/fold {list(chosen.values())} | all {bd['overall']['score']:.3f} | NEXAR {n['score']:.3f} "
                  f"E {n['entry_acc']:.2f} C {n['collision_acc']:.2f} Ccat {n['collision_catastrophic']:.2f} | non-NEXAR {bd['source:non-NEXAR']['score']:.3f}", flush=True)
    import json as _j
    prev = _j.loads((R / "decode_motion.json").read_text()) if (R / "decode_motion.json").exists() else {}
    C.dump(R / "decode_motion.json", {**prev, **out})


if __name__ == "__main__": main()
