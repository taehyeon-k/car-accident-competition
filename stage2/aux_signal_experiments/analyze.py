"""Aggregate complementary-signal runs against the exact matched control (phase_study A0_d1 = B0, 8 CV seeds).

Adds to phase_study.analyze: Recall@1/2/4/8 of the direct heads (greedy-NMS peaks, official 0.3 s hit, FPS only offline),
parameter / runtime cost, and for boundary arms a local re-rank decoder:
  top-K direct ENTRY x top-K direct COLLISION pairs (e <= c), score = direct pair log-prob + w * (log sigma(b_E(i)) + log sigma(b_C(j))),
  K in {4, 8}; w chosen per fold on the OTHER folds from {0.25, 0.5, 1, 2}. No whole-video decoding.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from stage2.long_context_v2_experiments import common as C
from stage2.long_context_v2_experiments.diagnostics import recall_at_k, summarize_recall
from stage2.phase_study import analyze as PA

R = C.REPO / "stage2/aux_signal_experiments/results"
CTRL_ROOT = C.REPO / "stage2/phase_study/results"


GEN_ROOT = C.REPO / "stage2/generalization/results"


def root_of(run): return R if (R / run).is_dir() else (GEN_ROOT if (GEN_ROOT / run).is_dir() else CTRL_ROOT)


def logsig(x): x = np.asarray(x, float); return -np.logaddexp(0, -x)


def rerank(p, w, k):
    el, cl = PA.logp(p["entry_logits"]), PA.logp(p["collision_logits"])
    be, bc = logsig(p["bnd_entry_logits"]), logsig(p["bnd_collision_logits"])
    best, arg = -np.inf, None
    for i in np.argsort(-el)[:k]:
        for j in np.argsort(-cl)[:k]:
            if i <= j:
                s = el[i] + cl[j] + w * (be[i] + bc[j])
                if s > best: best, arg = s, (i, j)
    if arg is None: arg = PA.decode_pair(p, "D0")
    q = {kk: v for kk, v in p.items() if kk not in ("entry_logits", "collision_logits", "bnd_entry_logits", "bnd_collision_logits",
                                                   "risk_logits", "lane_logits", "phase_logp", "frames")}
    q["entry_frame"], q["collision_frame"] = int(p["frames"][int(arg[0])]), int(p["frames"][int(arg[1])])
    return q


def rerank_cv(folds, k):
    ws = (0.25, 0.5, 1.0, 2.0)
    cache = {(f, w): [rerank(p, w, k) for p in folds[f]] for f in folds for w in ws}
    out, chosen = [], []
    for f in folds:
        w = max(ws, key=lambda w: C.metrics([q for g in folds if g != f for q in cache[(g, w)]])["score"])
        chosen.append(w); out += cache[(f, w)]
    return out, chosen


def recall_block(preds):
    s = summarize_recall(recall_at_k(preds, 8)); out = {}
    for key, short in (("overall", "all"), ("source:NEXAR", "NEXAR")):
        for e in ("entry", "collision"):
            for k in (1, 2, 4, 8): out[f"{short}.{e}_R@{k}"] = s[key][f"{e}_R@{k}"]
    return out


def summarize(run):
    root = root_of(run)
    res = PA.summarize(run, decoders=False, root=root)
    _, cv, metrics = PA.load_run(run, root)
    pooled = {s: [p for k in range(5) for p in cv[s][k]] for s in cv}
    res["recall"] = PA.mean_std([recall_block(p) for p in pooled.values()])
    res["params"] = metrics[0]["runtime"]["parameter_count"]
    res["head_ms"] = float(np.mean([m["runtime"].get("head_ms_per_video", np.nan) for m in metrics]))
    first = pooled[next(iter(pooled))][0]
    if "bnd_entry_logits" in first:
        dec = {}
        for k in (4, 8):
            runs = [rerank_cv(cv[s], k) for s in cv]
            dec[f"bnd_rerank_K{k}"] = PA.mean_std([PA.flat(r[0]) for r in runs]); dec[f"bnd_rerank_K{k}"]["_w"] = [r[1] for r in runs]
        res["decoders"] = dec
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--control", default="A0_d1")
    ap.add_argument("--out", default=None); a = ap.parse_args()
    results = []
    for run in a.runs:
        r = summarize(run)
        if run != a.control: r["paired"] = PA.paired(run, a.control, root=root_of(run), control_root=root_of(a.control))
        results.append(r)
    print(PA.table(results, a.control))
    print("\n| run | params | E R@1/2/4/8 | C R@1/2/4/8 | NEXAR C R@1/4 | Δ E hits | Δ C hits | top-10-clip share |")
    print("|---|---|---|---|---|---|---|---|")
    for r in results:
        rc = r["recall"]; g = lambda k: rc[k][0]; pr = r.get("paired", {})
        print(f"| {r['run']} | {r['params']} | " + "/".join(f"{g(f'all.entry_R@{k}'):.3f}" for k in (1, 2, 4, 8)) + " | "
              + "/".join(f"{g(f'all.collision_R@{k}'):.3f}" for k in (1, 2, 4, 8)) + f" | {g('NEXAR.collision_R@1'):.3f}/{g('NEXAR.collision_R@4'):.3f} | "
              + (f"{pr['delta_entry_hits']:+.1f} | {pr['delta_collision_hits']:+.1f} | {pr['share_of_change_in_top10_clips']:.2f} |" if pr else "| | |"))
    for r in results:
        for name, m in r.get("decoders", {}).items():
            print(f"{r['run']} {name}: all {PA.fmt(m, 'all.score')} (D0 {PA.fmt(r['cv'], 'all.score')}) NEXAR {PA.fmt(m, 'NEXAR.score')} "
                  f">1000 {PA.fmt(m, '>1000.score')} w={m['_w']}")
    if a.out: C.dump(a.out, results)


if __name__ == "__main__": main()
