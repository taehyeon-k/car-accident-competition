"""Leaderboard forecast for a Stage 2 candidate (methodology fixed 2026-09-29; see reports/stage2_lb_forecast_methodology.md).

Validation = the 1/3-frame-rate CV condition (native frames[::3], adaptive re-sampling, motion recomputed; robust_eval.item k=3),
duplicate-clean 284 clips, 5-fold out-of-fold, plain decoding (probability average + constrained anchors).
Calibration = the pool-free scored recipes (no MM-AU/CCD/AIHUB unlabelled pool): v5 .5314, v6 .5277, v7 .5464, v8 .59293,
v8kde .5886; LB = a + b * S_third (leave-one-out MAE .0048, leave-two-out .0069).
Forecast for a candidate = LB(v8) + b * (S_third(candidate) - S_third(v8)), both at the SAME seeds (anchored on v8, so seed count /
ensemble size effects cancel), with a paired bootstrap over clips for the uncertainty of the difference.
Valid only for recipes inside the calibration domain: labelled clips only, no unlabelled pool (recipes trained with the pool scored
~.03 below this forecast). Decision rule: a candidate beats v8 only if the forecast gain exceeds 2 bootstrap SDs.
Usage: python -m stage2.generalization.lb_predict CAND_KEY [CAND_KEY ...] --ref E4_sa+E2_sa+XN4_sa  (entry_suite keys at matched seeds)
"""
from __future__ import annotations

import argparse, json

import numpy as np

from stage2.long_context_v2_experiments import common as C

S = C.REPO / "stage2/generalization/results/entry_suite"
LBS = {"v5": .5314, "v6": .5277, "v7": .5464, "v8": .59293, "v8kde": .5886}  # pool-free scored recipes (Stage-2-only LB)
_ROWS = json.loads((C.REPO / "stage2/generalization/results/lb_forecast_rows.json").read_text())  # lb_forecast.py (seeds 0-1 proxies)
CAL = {k: (_ROWS[k]["third"], v) for k, v in LBS.items()}
LB_V8 = .59293


def slope():
    x = np.array([v[0] for v in CAL.values()]); y = np.array([v[1] for v in CAL.values()])
    return float(np.polyfit(x, y, 1)[0])


def third(key):
    return {p["sample_id"]: p for p in json.loads((S / f"{key}.json").read_text())["preds"]["k3"]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cands", nargs="+"); ap.add_argument("--ref", default="E4_sa+E2_sa+XN4_sa")
    ap.add_argument("--boot", type=int, default=500); a = ap.parse_args(); b = slope(); ref = third(a.ref); rng = np.random.default_rng(0)
    print(f"slope b = {b:.3f} (LB per 1/3-rate CV point); reference {a.ref}: S_third {C.metrics(list(ref.values()))['score']:.4f} -> LB {LB_V8}")
    for key in a.cands:
        cand = third(key); ids = sorted(set(cand) & set(ref)); d0 = C.metrics([cand[i] for i in ids])["score"] - C.metrics([ref[i] for i in ids])["score"]
        ds = []
        for _ in range(a.boot):
            s = rng.choice(ids, len(ids)); ds.append(C.metrics([cand[i] for i in s])["score"] - C.metrics([ref[i] for i in s])["score"])
        sd = float(np.std(ds)); fc = LB_V8 + b * d0
        verdict = "likely better than v8" if b * d0 > 2 * b * sd else ("likely worse than v8" if b * d0 < -2 * b * sd else "indistinguishable from v8")
        print(f"{key:36s} dS_third {d0:+.4f} (paired SD {sd:.4f}) -> forecast LB {fc:.4f} +- {b * sd:.4f} (1 SD) -> {verdict}")


if __name__ == "__main__": main()
