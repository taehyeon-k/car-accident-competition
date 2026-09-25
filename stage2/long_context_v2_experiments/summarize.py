"""Aggregate runs: fixed-split mean±std over seeds, and pooled 5-fold CV out-of-fold predictions."""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np

from . import common as C

R = C.REPO / "stage2/long_context_v2_experiments/results"
KEYS = [("overall", "all"), ("source:NEXAR", "NEXAR"), ("source:non-NEXAR", "nonNEX"), ("bin:>1000", ">1000"),
        ("source:MMAU", "MMAU"), ("source:AIHUB", "AIHUB"), ("source:CCD", "CCD")]


def fixed(run):
    ms = [json.loads(p.read_text()) for p in sorted((R / run).glob("seed*/metrics.json"))]
    if not ms: return None
    out = {"n_seeds": len(ms)}
    for key, short in KEYS:
        for f in ("score", "entry_acc", "collision_acc", "entry_catastrophic", "collision_catastrophic"):
            v = [m["breakdown"][key][f] for m in ms if m["breakdown"].get(key, {}).get("n")]
            out[f"{short}.{f}"] = (float(np.mean(v)), float(np.std(v))) if v else None
    for e in ("entry", "collision"):
        for k in (1, 4, 8):
            v = [m["recall_at_k"]["source:NEXAR"][f"{e}_R@{k}"] for m in ms]
            out[f"NEXAR.{e}_R@{k}"] = (float(np.mean(v)), float(np.std(v)))
    out["best_epochs"] = [m["best_epoch"] for m in ms]
    out["train_minus_val"] = float(np.mean([m["train_fpsblind_at_best"] - m["val_fpsblind_at_best"] for m in ms]))
    out["params"] = ms[0]["runtime"]["parameter_count"]
    return out


def cv(run):
    by_seed = {}
    for p in sorted((R / run / "cv").glob("fold*_seed*/predictions.json")):
        seed = p.parent.name.split("_seed")[1]
        by_seed.setdefault(seed, []).extend(json.loads(p.read_text()))
    by_seed = {s: v for s, v in by_seed.items() if len(v) == 349}
    if not by_seed: return None
    out = {"n_seeds": len(by_seed)}
    rng = np.random.default_rng(0)
    for key, short in KEYS:
        vals = [C.breakdown(v)[key] for v in by_seed.values()]
        for f in ("score", "entry_acc", "collision_acc", "entry_catastrophic", "collision_catastrophic"):
            x = [m[f] for m in vals]
            out[f"{short}.{f}"] = (float(np.mean(x)), float(np.std(x)))
    nex = [p for p in next(iter(by_seed.values())) if C.source(p) == "NEXAR"]
    bs = [C.metrics([nex[i] for i in rng.integers(0, len(nex), len(nex))])["score"] for _ in range(1000)]
    out["NEXAR.score_ci_seed_first"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return out


def fmt(t):
    return "   -   " if t is None else f"{t[0]:.3f}±{t[1]:.3f}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="*"); a = ap.parse_args()
    runs = a.runs or sorted(p.name for p in R.iterdir() if p.is_dir() and p.name != "diagnostics")
    table = {}
    print(f"{'run':28s} | fixed split (mean±std over seeds)                                   | 5-fold CV pooled (349 clips, NEXAR 80)")
    print(f"{'':28s} | {'all':11s} {'NEXAR':11s} {'nonNEX':11s} {'NEX_E':11s} {'NEX_C':11s} | {'all':11s} {'NEXAR':11s} {'nonNEX':11s} {'NEX_E':11s} {'NEX_C':11s} {'NEX_Ccat':11s}")
    for run in runs:
        f, c = fixed(run), cv(run)
        table[run] = {"fixed": f, "cv": c}
        fs = lambda k: fmt(f.get(k)) if f else "   -   "
        cs = lambda k: fmt(c.get(k)) if c else "   -   "
        print(f"{run:28s} | {fs('all.score')} {fs('NEXAR.score')} {fs('nonNEX.score')} {fs('NEXAR.entry_acc')} {fs('NEXAR.collision_acc')} | "
              f"{cs('all.score')} {cs('NEXAR.score')} {cs('nonNEX.score')} {cs('NEXAR.entry_acc')} {cs('NEXAR.collision_acc')} {cs('NEXAR.collision_catastrophic')}"
              + (f"  seeds={f['n_seeds']}" if f else "") + (f" cv_seeds={c['n_seeds']}" if c else ""))
    C.dump(R / "summary_table.json", table)


if __name__ == "__main__": main()
