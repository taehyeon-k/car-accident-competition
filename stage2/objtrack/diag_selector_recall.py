"""Narrowing campaign, Experiment B diagnostic — oracle-track recall@K of candidate selectors (oracle from diag_selection.json).
Selectors over the 6-slot prominence candidates (cache_objcrop6): PROM = area x centrality order, SEL-H = causal relevance
(objlane_features.relevance). Recall = the oracle track is among the first K slots at any native frame within ENTRY +- 2 frames."""
from __future__ import annotations

import json, os

os.environ.setdefault("OBJ_K_IN", "6")
import numpy as np

from stage2.long_context_v2_experiments import common as C
from stage2.objtrack import objlane_features as O


def main():
    recs = json.loads((C.REPO / "stage2/generalization/results/diag_selection.json").read_text()); pca = dict(np.load(O.PCA))
    rows = {r["sample_id"]: r for r in C.rows("all")}; res = []
    for x in recs:
        if x["oracle"] is None or not (O.CROPS / f"{x['sid']}.npz").exists(): continue
        d = np.load(O.CROPS / f"{x['sid']}.npz"); corr = np.load(O.CORR / f"{x['sid']}.npz"); e = int(rows[x["sid"]]["entry_frame"])
        _, oh = O.clip_features(d, corr, 1, "full", pca, select="heur", k_out=6, return_order=True)
        _, op = O.clip_features(d, corr, 1, "full", pca, select="none", k_out=6, return_order=True)
        fr = range(max(0, e - 2), min(len(oh), e + 3))
        rk = lambda od: min([od[f].index(x["oracle"]) + 1 for f in fr if x["oracle"] in od[f]] or [99])
        res.append({"sid": x["sid"], "source": x["source"], "gap": x["gap_s"], "prom": rk(op), "selh": rk(oh), "OB_D_hit": x["OB_D_hit"], "E4_hit": x["E4_hit"]})
    C.dump(C.REPO / "stage2/generalization/results/diag_selector_recall.json", res)
    rec = lambda xs, k, K: round(float(np.mean([y[k] <= K for y in xs])), 3) if xs else None
    print(f"n {len(res)}  recall@1/3/6  PROM {[rec(res,'prom',K) for K in (1,3,6)]}  SEL-H {[rec(res,'selh',K) for K in (1,3,6)]}")
    for s in ("AIHUB", "CCD", "MMAU", "NEXAR"):
        xs = [y for y in res if y["source"] == s]; print(f"  {s:6s} n {len(xs):3d} @3 PROM {rec(xs,'prom',3)} SEL-H {rec(xs,'selh',3)}")
    xs = [y for y in res if y["gap"] >= 1.5]; print(f"  gap>=1.5 n {len(xs)} @3 PROM {rec(xs,'prom',3)} SEL-H {rec(xs,'selh',3)}")
    xs = [y for y in res if not y["OB_D_hit"] and y["E4_hit"]]; print(f"  OB_D-hurts n {len(xs)} @3 PROM {rec(xs,'prom',3)} SEL-H {rec(xs,'selh',3)}")


if __name__ == "__main__": main()
