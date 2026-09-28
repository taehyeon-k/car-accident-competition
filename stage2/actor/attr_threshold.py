"""Deepening Exp 1 — attribute decision thresholds (evasion, side) chosen by NESTED CV on out-of-fold probabilities.

For each CV fold f: threshold t_f = argmax over a grid of the macro F1 of the OTHER four folds' native-condition predictions; applied
to fold f under every condition (native, 1/2, 1/3, crops, DACON-like view). Reports F1 and score with the 0.5 threshold vs the nested
threshold, the per-fold thresholds (stability) and the threshold fitted on all clips (deployment value).
Usage: python -m stage2.actor.attr_threshold ARM_KEY [ARM_KEY ...]   (suite result keys with _pev / _psr, e.g. ...@s012t)
"""
from __future__ import annotations

import json, sys

import numpy as np

from stage2.long_context_v2_experiments import common as C

S = C.REPO / "stage2/generalization/results/entry_suite"; FOLDS = C.REPO / "stage2/long_context_v2_experiments/folds"
GRID = np.round(np.arange(0.30, 0.801, 0.02), 2)
ATTR = {"evasion": ("_pev", "evasion_gt", "evasion_space"), "side": ("_psr", "entry_side_gt", "entry_side")}


def best_t(ps, key, gt):
    return max(GRID, key=lambda t: (C.macro_f1([q[gt] for q in ps], [int(q[key] >= t) for q in ps]), -abs(t - .5)))


def main():
    fold_of = {r["sample_id"]: f for f in range(5) for r in C.rows(str(FOLDS / f"fold{f}_val.jsonl"))}
    for arm in sys.argv[1:]:
        P = json.loads((S / f"{arm}.json").read_text())["preds"]; print(f"## {arm}")
        for name, (key, gt, dec) in ATTR.items():
            ts = {f: best_t([q for q in P["k1"] if fold_of[q["sample_id"]] != f], key, gt) for f in range(5)}
            t_all = best_t(P["k1"], key, gt)
            print(f"  {name}: nested thresholds per fold {[float(ts[f]) for f in range(5)]} | fitted on all clips {float(t_all)}")
            for c in P:
                base = C.metrics(P[c]); new = [dict(q, **{dec: int(q[key] >= ts[fold_of[q["sample_id"]]])}) for q in P[c]]; nm = C.metrics(new)
                f1k = "evasion_f1" if name == "evasion" else "side_f1"
                print(f"    {c:7s} F1 {base[f1k]:.3f} -> {nm[f1k]:.3f} ({nm[f1k] - base[f1k]:+.3f}) | score {base['score']:.4f} -> {nm['score']:.4f} ({nm['score'] - base['score']:+.4f})")


if __name__ == "__main__": main()
