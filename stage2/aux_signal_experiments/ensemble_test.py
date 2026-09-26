"""Ensemble complementarity (Exp 5B) on 5-fold OOF predictions with the unchanged v5 decoding
(camera-shift COLLISION fusion, beta per fold chosen on the other folds, native snap, no prior).

B3 = v5 as built: C0_avg + X_ema + M_motion (all CV seeds) + PH_phase CV seeds {0,1}  (8 members / fold).
  add     : B3 + candidate family (2-seed draw)            vs  B3 + same-size control draw (A0_d1 / PH_repro_structsel)
  replace : B3 with M_motion replaced by candidate draw     vs  B3 with M_motion replaced by control draw (motion variants)
Draws: disjoint CV-seed pairs of the added family, e.g. {0,1} {2,3} {4,5} {6,7}; reported as mean +- std over draws.
Diversity vs B3 (per draw): hit disagreement rate, Spearman correlation of per-video normalized |error| (ENTRY, COLLISION),
oracle gain = mean over clips of max(hits_B3, hits_cand) - hits_B3 (the candidate decoded alone, direct decoding).
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy.stats import spearmanr

from stage2.long_context_v2_experiments import common as C
from stage2.long_context_v2_experiments.decode_motion import decode
from stage2.phase_study import ensemble_test as PE

ROOTS = [C.REPO / "stage2/long_context_v2_experiments/results", C.REPO / "stage2/phase_study/results",
         C.REPO / "stage2/aux_signal_experiments/results"]
PE.root = lambda run: next(r for r in ROOTS if (r / run).is_dir())  # members_fold resolves runs through PE.root


def members(spec, k):
    """spec = [(run, seeds or None)] -> members for fold k."""
    return [m for run, seeds in spec for m in PE.members_fold(run, k, seeds)]


def pooled(spec):
    out = {}
    for k in range(5):
        mem = members(spec, k); merged = []
        for sid in mem[0][0]:
            items = [m[0][sid] for m in mem]
            p = {kk: v for kk, v in items[0].items() if not kk.endswith("_logits") and kk != "phase_logp"}
            for e in ("entry", "collision"):
                p[f"{e}_logp"] = np.log(np.mean([np.exp(PE.lp(it[f"{e}_logits"])) for it in items], 0) + 1e-12)
            p["entry_side"] = int(np.mean([m[1][sid][0] for m in mem]) >= .5)
            p["evasion_space"] = int(np.mean([m[1][sid][1] for m in mem]) >= .5)
            merged.append(p)
        out[k] = merged
    return out


def fused_cv(spec):
    folds = pooled(spec); cache = {}
    def preds(k, b):
        if (k, b) not in cache: cache[(k, b)] = [decode(p, None, 0, 0, b, False, True) for p in folds[k]]
        return cache[(k, b)]
    final, chosen = [], []
    for k in folds:
        def other(b):
            ps = [q for j in folds if j != k for q in preds(j, b)]
            return C.metrics([q for q in ps if C.source(q) == "NEXAR"])["collision_acc"] + C.metrics(ps)["score"]
        b = max(PE.BETAS, key=other); chosen.append(b); final += preds(k, b)
    return final, chosen


def plain_cv(spec):
    return [decode(p, None, 0, 0, 0.0, False, False) for k, ps in pooled(spec).items() for p in ps]


def per_clip(preds):
    fps = C.fps_table(); out = {}
    for p in preds:
        n = max(p["num_available_frames"] - 1, 1)
        out[p["sample_id"]] = {e: (abs(p[f"{e}_frame"] - p[f"{e}_gt"]) / fps[p["sample_id"]] <= .300001,
                                   abs(p[f"{e}_frame"] - p[f"{e}_gt"]) / n) for e in ("entry", "collision")}
    return out


def diversity(base_preds, cand_preds):
    a, b = per_clip(base_preds), per_clip(cand_preds); ids = sorted(a)
    res = {}
    for e in ("entry", "collision"):
        ha = np.array([a[i][e][0] for i in ids], float); hb = np.array([b[i][e][0] for i in ids], float)
        ea = np.array([a[i][e][1] for i in ids]); eb = np.array([b[i][e][1] for i in ids])
        res[f"{e}_disagree"] = float((ha != hb).mean()); res[f"{e}_err_spearman"] = float(spearmanr(ea, eb)[0])
        res[f"{e}_oracle_gain"] = float((np.maximum(ha, hb) - ha).mean())
    return res


def row(name, b):
    o, n = b["overall"], b["source:NEXAR"]
    return {"name": name, "all": o["score"], "E": o["entry_acc"], "C": o["collision_acc"], "side": o["side_f1"], "eva": o["evasion_f1"],
            "NEXAR": n["score"], ">1000": b["bin:>1000"]["score"], "nonNEX": b["source:non-NEXAR"]["score"],
            "Ecat": o["entry_catastrophic"], "Ccat": o["collision_catastrophic"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("families", nargs="+", help="candidate / control run ids")
    ap.add_argument("--mode", choices=["add", "replace"], default="add")
    ap.add_argument("--seed-sets", default="0,1 2,3 4,5 6,7"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    base = [("C0_avg", None), ("X_ema", None), ("M_motion", None), ("PH_phase", {0, 1})]
    b3, _ = fused_cv(base); b3_row = row("B3 (v5 as built)", C.breakdown(b3))
    print(f"B3 all {b3_row['all']:.4f} NEXAR {b3_row['NEXAR']:.4f} >1000 {b3_row['>1000']:.4f}", flush=True)
    res = {"B3": b3_row}
    draws = [set(int(x) for x in d.split(",")) for d in a.seed_sets.split()]
    for fam in a.families:
        rows, divs = [], []
        for d in draws:
            if not all(PE.members_fold(fam, 0, {s}) for s in d): continue
            spec = ([x for x in base if x[0] != "M_motion"] if a.mode == "replace" else base) + [(fam, d)]
            fused, betas = fused_cv(spec)
            rows.append(row(f"{fam} {sorted(d)}", C.breakdown(fused)))
            divs.append(diversity(b3, plain_cv([(fam, d)])))
            print(f"  {a.mode} {fam} seeds {sorted(d)}: all {rows[-1]['all']:.4f} E {rows[-1]['E']:.3f} C {rows[-1]['C']:.3f} "
                  f"NEXAR {rows[-1]['NEXAR']:.4f} >1000 {rows[-1]['>1000']:.4f} beta {betas}", flush=True)
        if not rows: continue
        agg = {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))) for k in rows[0] if k != "name"}
        dv = {k: float(np.mean([d_[k] for d_ in divs])) for k in divs[0]}
        res[fam] = {"draws": rows, "mean": agg, "diversity_vs_B3": dv}
        print(f"==> {a.mode} {fam}: all {agg['all'][0]:.4f}±{agg['all'][1]:.4f} NEXAR {agg['NEXAR'][0]:.4f} >1000 {agg['>1000'][0]:.4f} "
              f"E {agg['E'][0]:.3f} C {agg['C'][0]:.3f} | diversity {dict((k, round(v, 3)) for k, v in dv.items())}", flush=True)
    if a.out: C.dump(a.out, res)


if __name__ == "__main__": main()
