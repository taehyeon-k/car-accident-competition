"""Leaderboard-forecast methodology check (2026-09-29) on the Stage-2-only leaderboard scores.

Rows = scored recipes: CV proxies (seeds 0-1, plain decoding, 284 duplicate-clean clips) from robustness_profile.json, plus v8kde from
entry_suite (KDe+E2_sa+XN4_sa@s01). Provenance per recipe is read from its members' training configs: `pool` = any member trained with
the MM-AU/CCD/AIHUB unlabelled pool (extra_unl or unl_consistency); `nexar` = NEXAR pseudo-labelled extras.
Candidate forecasters are evaluated by leave-one-out and leave-two-out (every pair) MAE, on (a) all recipes and (b) the pool-free
recipes only. Also reports the CV noise of each feature (paired bootstrap over clips) so forecast differences can be judged.
Usage: python -m stage2.generalization.lb_forecast
"""
from __future__ import annotations

import itertools, json

import numpy as np
import torch

from stage2.long_context_v2_experiments import common as C
from .robustness_analyze import excluded
from .robustness_profile import RECIPES

LB = {"v5": .5314, "v6": .5277, "v7": .5464, "v8": .59293, "v10": .5590, "v12": .5635, "v13": .5609, "v8kde": .5886}
FAMS = {**{k: sorted(set(v["entry"] + v["coll"] + v["attr"])) for k, v in RECIPES.items()}, "v8kde": ["KDe", "E2_sa", "XN4_sa"]}
ROOTS = [C.REPO / "stage2/generalization/results", C.REPO / "stage2/aux_signal_experiments/results", C.REPO / "stage2/long_context_v2_experiments/results",
         C.REPO / "stage2/phase_study/results"]
CONDS = {"native": ("k1_crop0.0", "k1"), "half": ("k2_crop0.0", "k2"), "third": ("k3_crop0.0", "k3"), "crop50": ("k1_crop0.5", "crop50"), "crop25": ("k1_crop0.25", "crop25")}


def provenance(rec):
    pool = nexar = False
    for fam in FAMS[rec]:
        ck = next((r / fam / "cv" / "fold0_seed0" / "checkpoint.pt" for r in ROOTS if (r / fam / "cv" / "fold0_seed0" / "checkpoint.pt").exists()), None)
        if ck is None: continue
        cfg = torch.load(ck, map_location="cpu", weights_only=False).get("config", {})
        pool |= bool(cfg.get("extra_unl")) or float(cfg.get("unl_consistency") or 0) > 0
        nexar |= bool(cfg.get("extra_nexar"))
    return pool, nexar


def preds(rec):
    EX = excluded()
    if rec == "v8kde":
        P = json.loads((C.REPO / "stage2/generalization/results/entry_suite/KDe+E2_sa+XN4_sa@s01.json").read_text())["preds"]
        return {k: [p for p in P[v[1]] if p["sample_id"] not in EX] for k, v in CONDS.items()}
    P = json.loads((C.REPO / "stage2/generalization/results/robustness_profile.json").read_text())[rec]
    return {k: [p for p in P[v[0]] if p["sample_id"] not in EX] for k, v in CONDS.items()}


def loo(X, y, pairs=False):
    n = len(y); errs = []
    for held in (itertools.combinations(range(n), 2) if pairs else ((i,) for i in range(n))):
        tr = [i for i in range(n) if i not in held]
        A = np.c_[np.ones(len(tr)), X[tr]]; w = np.linalg.lstsq(A, y[tr], rcond=None)[0]
        errs += [abs(np.r_[1, X[i]] @ w - y[i]) for i in held]
    return float(np.mean(errs)), float(np.max(errs))


def main():
    rows = {}
    for rec in LB:
        P = preds(rec); m = {k: C.metrics(v) for k, v in P.items()}; pool, nexar = provenance(rec)
        rows[rec] = {"pool": float(pool), "nexar": float(nexar), **{f"{k}": m[k]["score"] for k in CONDS}, **{f"{k}_E": m[k]["entry_acc"] for k in CONDS},
                     **{f"{k}_C": m[k]["collision_acc"] for k in CONDS}}
    names = list(LB); y = np.array([LB[r] for r in names])
    print("| recipe | LB | pool | nexar | native | 1/2 | 1/3 | crop50 | crop25 | ENTRY 1/3 | crop25 ENTRY |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for r in names:
        v = rows[r]
        print(f"| {r} | {LB[r]:.4f} | {int(v['pool'])} | {int(v['nexar'])} | {v['native']:.3f} | {v['half']:.3f} | {v['third']:.3f} | {v['crop50']:.3f} | {v['crop25']:.3f} | {v['third_E']:.3f} | {v['crop25_E']:.3f} |")
    cands = {"mean baseline": [], "native": ["native"], "1/2": ["half"], "1/3": ["third"], "mean(1/2,1/3)": ["m23"], "1/3 + crop25 ENTRY": ["third", "crop25_E"],
             "1/3 + pool flag": ["third", "pool"], "mean(1/2,1/3) + pool flag": ["m23", "pool"], "native + pool flag": ["native", "pool"]}
    for r in names: rows[r]["m23"] = (rows[r]["half"] + rows[r]["third"]) / 2
    for subset_name, subset in (("all 8 recipes", names), ("pool-free recipes (labelled-only rule)", [r for r in names if not rows[r]["pool"]])):
        idx = [names.index(r) for r in subset]; yy = y[idx]
        print(f"\n### {subset_name}: {', '.join(subset)}\n| forecaster | LOO MAE | LOO max | leave-2-out MAE | leave-2-out max |\n|---|---|---|---|---|")
        for cname, feats in cands.items():
            if "pool" in feats and subset_name.startswith("pool-free"): continue
            X = np.array([[rows[r][f] for f in feats] for r in subset]).reshape(len(subset), -1)
            if len(subset) - 2 <= X.shape[1]: continue
            a, b = loo(X, yy); c, d = loo(X, yy, pairs=True)
            print(f"| {cname} | {a:.4f} | {b:.4f} | {c:.4f} | {d:.4f} |")
    # CV noise: paired bootstrap of the 1/3 and 1/2 score difference v8kde - v8 over clips
    rng = np.random.default_rng(0); P8, PK = preds("v8"), preds("v8kde")
    for cond in ("half", "third", "native"):
        a = {p["sample_id"]: p for p in P8[cond]}; b = {p["sample_id"]: p for p in PK[cond]}; ids = sorted(set(a) & set(b)); d = []
        for _ in range(300):
            s = rng.choice(ids, len(ids)); d.append(C.metrics([b[i] for i in s])["score"] - C.metrics([a[i] for i in s])["score"])
        print(f"CV noise: v8kde - v8 {cond} score diff {np.mean(d):+.4f}, paired-bootstrap SD {np.std(d):.4f}")
    C.dump(C.REPO / "stage2/generalization/results/lb_forecast_rows.json", rows)


if __name__ == "__main__": main()
