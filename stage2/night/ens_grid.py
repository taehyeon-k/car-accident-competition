"""Night campaign — event-specific family selection over cached out-of-fold probabilities (generalization/ens_cache, seeds 0-1,
full positions). ENTRY families = any non-empty subset of the listed families; COLLISION families from a few fixed sets; side /
evasion from ENTRY ∪ COLLISION families (as ens_search). Plain decoding (probability average + constrained anchors).
Reports native / 1/2 / 1/3 / crop50 / crop25 score and the LB two-factor forecast; ranks by forecast and by native score.
Usage: python -m stage2.night.ens_grid FAM [FAM ...] --coll "E4_sa,E2_sa,XN4_sa" "KDe,KDe_E2,KDe_XN4"
"""
from __future__ import annotations

import argparse, itertools

from stage2.long_context_v2_experiments import common as C
from stage2.generalization.ens_search import decode, forecast_model

import numpy as np

CONDS = ("k1_crop0.0", "k2_crop0.0", "k3_crop0.0", "k1_crop0.5", "k1_crop0.25")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("fams", nargs="+"); ap.add_argument("--coll", nargs="+", required=True)
    ap.add_argument("--top", type=int, default=12); a = ap.parse_args(); W = forecast_model(); res = []
    colls = [tuple(c.split(",")) for c in a.coll]
    for r in range(1, len(a.fams) + 1):
        for ent in itertools.combinations(a.fams, r):
            for coll in colls:
                m = {c: C.metrics(decode(list(ent), list(coll), 0.0, c)) for c in CONDS}
                fc = float(W @ [1, m["k3_crop0.0"]["score"], m["k1_crop0.25"]["entry_acc"]])
                res.append((ent, coll, [m[c]["score"] for c in CONDS], m["k1_crop0.0"]["entry_acc"], m["k1_crop0.0"]["collision_acc"], fc))
    for key, name in ((5, "forecast"), (2, "native score")):
        print(f"\n### top {a.top} by {name}\n| ENTRY families | COLLISION families | score nat / ½ / ⅓ / crop50 / crop25 | ENTRY nat | COLL nat | forecast |\n|---|---|---|---|---|---|")
        srt = sorted(res, key=(lambda x: -x[5]) if key == 5 else (lambda x: -x[2][0]))
        for ent, coll, sc, e, c, fc in srt[:a.top]:
            print(f"| {'+'.join(ent)} | {'+'.join(coll)} | " + " / ".join(f"{v:.3f}" for v in sc) + f" | {e:.3f} | {c:.3f} | {fc:.4f} |")


if __name__ == "__main__": main()
