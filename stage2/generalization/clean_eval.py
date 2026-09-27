"""Duplicate-clean re-scoring: run an existing evaluator with every metric computed WITHOUT the validation clips that
(a) have a near-duplicate labelled clip in another CV fold, or (b) have a near-duplicate in the expansion pool (dup_audit.py).
Usage: python -m stage2.generalization.clean_eval goal_eval|analyze|robust_eval ARGS...
"""
from __future__ import annotations

import json
import runpy
import sys

from stage2.long_context_v2_experiments import common as C

d = json.loads((C.REPO / "stage2/generalization/results/dup_audit.json").read_text())
EXCL = {t[0] for t in d["lab_vs_lab"] if t[2] >= 0.5 and t[4] != t[5]} | {t[1] for t in d["unl_vs_lab"] if t[2] >= 0.5}
_metrics = C.metrics


def metrics(preds):
    return _metrics([p for p in preds if p["sample_id"] not in EXCL])


C.metrics = metrics
if __name__ == "__main__":
    mod = {"goal_eval": "stage2.aux_signal_experiments.goal_eval", "analyze": "stage2.aux_signal_experiments.analyze",
           "robust_eval": "stage2.generalization.robust_eval"}[sys.argv[1]]
    print(f"[clean_eval] excluding {len(EXCL)} validation clips", flush=True)
    sys.argv = [mod] + sys.argv[2:]
    runpy.run_module(mod, run_name="__main__")
