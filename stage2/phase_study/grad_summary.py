"""Gradient interaction on the shared temporal-pyramid parameters (logged by train.py --grad-cos 8, fixed-split seeds).

For each run: mean over seeds of cos(g_direct, g_phase), cos(g_direct, g_transition), cos(g_phase, g_transition) and the
gradient-norm ratio |g_phase| / |g_direct| (unweighted losses; multiply by lambda for the effective ratio), in epoch windows.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from stage2.long_context_v2_experiments import common as C

R = C.REPO / "stage2/phase_study/results"
WINDOWS = [("init", 0, 0), ("ep1-3", 1, 3), ("ep4-10", 4, 10), ("ep11+", 11, 99)]
KEYS = ["direct~phase", "direct~transition", "phase~transition", "norm_ratio_phase/direct"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--out", default=None); a = ap.parse_args()
    res = {}
    print("| run | window | cos(direct, phase) | cos(direct, transition) | cos(phase, transition) | abs(g_phase)/abs(g_direct) |")
    print("|---|---|---|---|---|---|")
    for run in a.runs:
        hs = [json.loads(p.read_text()) for p in sorted((R / run).glob("seed*/history.json"))]
        hs = [h for h in hs if any("grad_cos" in e for e in h)]
        if not hs: continue
        res[run] = {}
        for name, lo, hi in WINDOWS:
            vals = {k: [e["grad_cos"][k] for h in hs for e in h if "grad_cos" in e and lo <= e["epoch"] <= hi and k in e["grad_cos"]] for k in KEYS}
            res[run][name] = {k: (float(np.mean(v)) if v else None) for k, v in vals.items()}
            f = lambda k: f"{res[run][name][k]:+.3f}" if res[run][name][k] is not None else "—"
            print(f"| {run} | {name} | {f('direct~phase')} | {f('direct~transition')} | {f('phase~transition')} | {f('norm_ratio_phase/direct')} |")
    if a.out: C.dump(a.out, res)


if __name__ == "__main__": main()
