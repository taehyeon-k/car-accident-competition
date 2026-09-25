"""Decode-time priors on saved out-of-fold CV logits (no training, no GPU).

For each CV fold k, priors are fitted on the *training folds* only, restricted to long clips (N > 500 frames; frame count
only, no FPS/source), and applied to fold-k clips with N > 500:
  pos   : Gaussian prior on normalized ENTRY / COLLISION position
  gap   : Gaussian prior on normalized ENTRY->COLLISION gap (joint pair decode)
Joint decode maximises  w*(log p_e(i) + log p_c(j)) + a*log N(pos_i) + a*log N(pos_j) + g*log N(pos_j - pos_i)  over i <= j.
NOTE: in this dataset every long clip is NEXAR, whose curated 40 s clips put the collision at ~50 %; the position prior
measures how much that curation artifact is worth. It transfers only if test clips are curated the same way.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np

from . import common as C

R = C.REPO / "stage2/long_context_v2_experiments/results"
LONG = 500


def oof(run):
    """OOF predictions per fold, probability-averaged over CV seeds (and over runs joined with '+')."""
    folds = {}
    for k in range(5):
        per_seed = [json.loads(p.read_text()) for r in run.split("+") for p in sorted((R / r / "cv").glob(f"fold{k}_seed*/predictions.json"))]
        full = [{x["sample_id"]: x for x in json.loads(p.read_text())} for r in run.split("+")
                for p in sorted((R / r / "cv").glob(f"fold{k}_seed*/oof_full.json"))]
        merged = []
        for items in zip(*per_seed):
            assert len({x["sample_id"] for x in items}) == 1
            p = dict(items[0])
            if len(full) == len(per_seed):  # ensemble attributes from probabilities
                sid = p["sample_id"]
                assert all(f[sid]["frames"] == p["frames"] for f in full)
                p["entry_side"] = int(np.mean([f[sid]["side_prob_right"] for f in full]) >= .5)
                p["evasion_space"] = int(np.mean([f[sid]["evasion_prob"] for f in full]) >= .5)
            for e in ("entry", "collision"):
                lp = [np.asarray(x[f"{e}_logits"]) - np.logaddexp.reduce(x[f"{e}_logits"]) for x in items]
                p[f"{e}_logp"] = np.log(np.mean([np.exp(x) for x in lp], 0) + 1e-12)
            merged.append(p)
        folds[k] = merged
    return folds


def fit(rows):
    rows = [r for r in rows if int(r["num_frames"]) > LONG]
    e = np.array([int(r["entry_frame"]) / (int(r["num_frames"]) - 1) for r in rows])
    c = np.array([int(r["collision_frame"]) / (int(r["num_frames"]) - 1) for r in rows])
    return {"e": (e.mean(), e.std() + 1e-3), "c": (c.mean(), c.std() + 1e-3), "g": ((c - e).mean(), (c - e).std() + 1e-3)}


def lognorm(x, ms):
    m, s = ms; return -0.5 * ((x - m) / s) ** 2


def decode(p, prior, a, g):
    fr = np.asarray(p["frames"], float); pos = (fr - fr[0]) / max(fr[-1] - fr[0], 1)
    se, sc = p["entry_logp"].copy(), p["collision_logp"].copy()
    if prior is not None and p["num_available_frames"] > LONG:
        se += a * lognorm(pos, prior["e"]); sc += a * lognorm(pos, prior["c"])
        pair = se[:, None] + sc[None, :] + (g * lognorm(pos[None, :] - pos[:, None], prior["g"]) if g else 0)
    else:
        pair = se[:, None] + sc[None, :]
    pair = np.where(np.triu(np.ones_like(pair, bool)), pair, -np.inf)
    i, j = np.unravel_index(np.argmax(pair), pair.shape)
    q = {k: v for k, v in p.items() if k not in ("entry_logits", "collision_logits", "entry_logp", "collision_logp", "frames")}
    q["entry_frame"], q["collision_frame"] = int(fr[i]), int(fr[j])
    return q


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); a_ = ap.parse_args()
    out = {}
    for run in a_.runs:
        folds = oof(run)
        settings = [("none", None, 0, 0)] + [(f"pos a={a}", "p", a, 0) for a in (0.25, 0.5, 1.0)] + \
                   [(f"pos a={a} + gap g={g}", "p", a, g) for a in (0.5, 1.0) for g in (0.5, 1.0)] + [("gap g=1 only", "p", 0, 1.0)]
        out[run] = {}
        print(f"\n== {run}  (CV OOF, {len(list((R / run / 'cv').glob('fold0_seed*/predictions.json')))} seed(s)/fold)")
        for name, use, a, g in settings:
            preds = []
            for k, items in folds.items():
                prior = fit(C.rows(str(C.REPO / f"stage2/long_context_v2_experiments/folds/fold{k}_train.jsonl"))) if use else None
                preds += [decode(p, prior, a, g) for p in items]
            b = C.breakdown(preds); out[run][name] = b
            n = b["source:NEXAR"]
            print(f"  {name:22s} all {b['overall']['score']:.3f} | NEXAR {n['score']:.3f} E {n['entry_acc']:.2f} C {n['collision_acc']:.2f} "
                  f"Ecat {n['entry_catastrophic']:.2f} Ccat {n['collision_catastrophic']:.2f} | non-NEXAR {b['source:non-NEXAR']['score']:.3f}")
    C.dump(R / "decode_priors.json", out)


if __name__ == "__main__": main()
