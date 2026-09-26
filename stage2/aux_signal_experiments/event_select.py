"""Event-specific ensemble: ENTRY and COLLISION distributions may come from different family subsets. CV only.

For each held-out fold k, using ONLY the other folds:
  1. rank families by their own pooled ENTRY accuracy and, separately, COLLISION accuracy (v5 decoding, beta 1)
  2. choose k_E, k_C (number of top-ranked families averaged for each event, member-count weighted) and the camera-shift beta
     by the other folds' overall score (decode_motion criterion: NEXAR COLLISION acc + overall score)
Attributes (--attr): side / evasion probability = mean over the top-k families ranked by their own other-fold macro-F1, and the
decision threshold in {0.35 .. 0.65}; k and the threshold chosen on the other folds by that attribute's macro-F1.
Decoding = v5 (motion fusion + native snap).
"""
from __future__ import annotations

import argparse, itertools

import numpy as np

from stage2.long_context_v2_experiments import common as C
from stage2.long_context_v2_experiments.decode_motion import decode
from . import ensemble_test as ET
from .goal_eval import parse, line

BETAS = [0.0, 0.5, 1.0, 2.0]


def family_tables(spec):
    """{family: {fold: {sid: (entry_prob, collision_prob, n_members)}}} plus a base pooled dict (attributes of all)."""
    fam = {}
    for run, seeds in spec:
        pooled = ET.pooled([(run, seeds)]); n = len(ET.members([(run, seeds)], 0))
        fam[run] = {k: {p["sample_id"]: (np.exp(p["entry_logp"]), np.exp(p["collision_logp"]), n, p["side_prob"], p["evasion_prob"])
                        for p in ps} for k, ps in pooled.items()}
    return fam, ET.pooled(spec)


def combine(base, fam, fe, fc):
    out = {}
    for k, ps in base.items():
        rows = []
        for p in ps:
            sid = p["sample_id"]; q = dict(p)
            for key, fs, i in (("entry_logp", fe, 0), ("collision_logp", fc, 1)):
                w = np.array([fam[f][k][sid][2] for f in fs], float)
                q[key] = np.log(sum(wi * fam[f][k][sid][i] for wi, f in zip(w, fs)) / w.sum() + 1e-12)
            rows.append(q)
        out[k] = rows
    return out


def attribute_select(final, fam, names):
    from stage2.spotting_experiments.train import macro_f1
    fold_of = {}
    for f in names:
        for k, d in fam[f].items():
            for sid in d: fold_of[sid] = k
    byk = {}
    for q in final: byk.setdefault(fold_of[q["sample_id"]], []).append(q)
    thr = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
    for attr, gt, idx, key in (("side", "entry_side_gt", 3, "entry_side"), ("evasion", "evasion_gt", 4, "evasion_space")):
        def prob(q, fs):
            k = fold_of[q["sample_id"]]; w = np.array([fam[f][k][q["sample_id"]][2] for f in fs], float)
            return float(sum(wi * fam[f][k][q["sample_id"]][idx] for wi, f in zip(w, fs)) / w.sum())
        for k in byk:
            others = [q for j in byk if j != k for q in byk[j]]
            f1 = lambda fs, t: macro_f1([q[gt] for q in others], [int(prob(q, fs) >= t) for q in others])
            rank = sorted(names, key=lambda f: -f1([f], 0.5))
            n, t = max(itertools.product(range(1, len(names) + 1), thr), key=lambda c: f1(rank[:c[0]], c[1]))
            print(f"  {attr} fold {k}: families {rank[:n]} threshold {t}")
            for q in byk[k]: q[key] = int(prob(q, rank[:n]) >= t)
    return [q for k in sorted(byk) for q in byk[k]]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("families", nargs="+"); ap.add_argument("--attr", action="store_true"); a = ap.parse_args()
    spec = parse(a.families); names = [r for r, _ in spec]
    fam, base = family_tables(spec)
    # single-family decoded predictions (beta 1) for ranking
    single = {f: {k: [decode(q, None, 0, 0, 1.0, False, True) for q in combine(base, fam, [f], [f])[k]] for k in base} for f in names}
    cache = {}
    def preds(k, fe, fc, b):
        key = (k, tuple(fe), tuple(fc), b)
        if key not in cache: cache[key] = [decode(q, None, 0, 0, b, False, True) for q in combine({k: base[k]}, fam, fe, fc)[k]]
        return cache[key]
    final, chosen = [], []
    for k in base:
        others = [j for j in base if j != k]
        acc = lambda f, e: np.mean([C.hit(q, e) for j in others for q in single[f][j]])
        rank_e = sorted(names, key=lambda f: -acc(f, "entry")); rank_c = sorted(names, key=lambda f: -acc(f, "collision"))
        def score(cfg):
            ke, kc, b = cfg; ps = [q for j in others for q in preds(j, rank_e[:ke], rank_c[:kc], b)]
            return C.metrics([q for q in ps if C.source(q) == "NEXAR"])["collision_acc"] + C.metrics(ps)["score"]
        cfg = max(itertools.product(range(1, len(names) + 1), range(1, len(names) + 1), BETAS), key=score)
        chosen.append((rank_e[:cfg[0]], rank_c[:cfg[1]], cfg[2])); final += preds(k, rank_e[:cfg[0]], rank_c[:cfg[1]], cfg[2])
    print(" + ".join(a.families)); print(line("CV-ES", C.breakdown(final), "| event-specific subsets"))
    if a.attr:
        final = attribute_select(final, fam, names); print(line("CV-ESA", C.breakdown(final), "| + attribute family/threshold selection"))
    for k, (fe, fc, b) in enumerate(chosen): print(f"  fold {k}: ENTRY from {fe} | COLLISION from {fc} | beta {b}")


if __name__ == "__main__": main()
