"""Matched-seed single-model CV comparison (fixes the seed-composition mismatch of analyze.py, which pools all seeds of each arm).

python -m stage2.generalization.matched_cv CONTROL CAND [CAND ...] --seeds 0 1 2 [--clean]
For each run and seed: pooled 5-fold OOF predictions (cv/fold*_seed{s}/predictions.json) -> official metrics. Reports the mean over
the SAME seeds for every run, and the paired delta vs CONTROL with a source-stratified clip bootstrap (2000 replicates; per replicate
the metrics - including macro-F1 - are recomputed on the resampled clips for every seed and averaged). --clean drops the validation
clips flagged by dup_audit.py (clean_eval.EXCL).
"""
from __future__ import annotations

import argparse, json
from collections import defaultdict

import numpy as np

from stage2.long_context_v2_experiments import common as C

ROOTS = [C.REPO / "stage2/generalization/results", C.REPO / "stage2/aux_signal_experiments/results", C.REPO / "stage2/phase_study/results"]
KEYS = ("score", "entry_acc", "collision_acc", "side_f1", "evasion_f1")


def load(run, seeds):
    root = next(r for r in ROOTS if (r / run / "cv").is_dir())
    out = {}
    for s in seeds:
        ps = [p for k in range(5) for p in json.loads((root / run / "cv" / f"fold{k}_seed{s}" / "predictions.json").read_text())]
        out[s] = {p["sample_id"]: p for p in ps}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--clean", action="store_true"); ap.add_argument("--boot", type=int, default=2000); a = ap.parse_args()
    excl = set()
    if a.clean:
        from stage2.generalization.clean_eval import EXCL; excl = EXCL
    data = {r: load(r, a.seeds) for r in a.runs}
    ids = sorted(set.intersection(*[set(d[s]) for d in data.values() for s in a.seeds]) - excl)
    by_src = defaultdict(list)
    for i in ids: by_src[C.source(data[a.runs[0]][a.seeds[0]][i])].append(i)
    def score(run, sample):
        ms = [C.metrics([data[run][s][i] for i in sample]) for s in a.seeds]
        return {k: float(np.mean([m[k] for m in ms])) for k in KEYS}
    base = {r: score(r, ids) for r in a.runs}
    nex = [i for i in ids if C.source(data[a.runs[0]][a.seeds[0]][i]) == "NEXAR"]
    rng = np.random.default_rng(0); boots = defaultdict(list)
    for _ in range(a.boot):
        sample = [i for s, lst in by_src.items() for i in rng.choice(lst, len(lst))]
        ctrl = score(a.runs[0], sample)["score"]
        for r in a.runs[1:]: boots[r].append(score(r, sample)["score"] - ctrl)
    print(f"seeds {a.seeds}, clips {len(ids)}{' (clean)' if a.clean else ''}")
    print("| run | CV | NEXAR | ENTRY | COLLISION | side | evasion | Δ vs control [95% CI] |\n|---|---|---|---|---|---|---|---|")
    for r in a.runs:
        b = base[r]; nx = score(r, nex)["score"]
        d = "" if r == a.runs[0] else f"{b['score'] - base[a.runs[0]]['score']:+.4f} [{np.percentile(boots[r], 2.5):+.4f}, {np.percentile(boots[r], 97.5):+.4f}]"
        print(f"| {r} | {b['score']:.4f} | {nx:.4f} | {b['entry_acc']:.3f} | {b['collision_acc']:.3f} | {b['side_f1']:.3f} | {b['evasion_f1']:.3f} | {d} |")


if __name__ == "__main__": main()
