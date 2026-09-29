"""Duplicate-clean fold0 rate and source screen for source-adversarial training."""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import torch

from stage2.long_context_v2_experiments import common as C
from stage2.generalization import robust_eval as R
from stage2.generalization.clean_eval import EXCL
from stage2.aux_signal_experiments.model import load as load_standard
from stage2.robust_goal_20260927.source_invariant import load as load_source


def main():
    root = C.REPO
    models = {
        "source_grl": load_source(root / "stage2/robust_goal_20260927/source_invariant/cv/fold0_seed0/checkpoint.pt", "cuda"),
        "E4_sa": load_standard(root / "stage2/generalization/results/E4_sa/cv/fold0_seed0/checkpoint.pt", "cuda"),
    }
    rows = [r for r in C.rows(str(R.FOLDS / "fold0_val.jsonl")) if r["sample_id"] not in EXCL]
    fps = C.fps_table()
    scores = {name: {} for name in models}
    for stride in (1, 2, 3):
        preds = defaultdict(list)
        for row in rows:
            sample = R.item(row, stride)
            for name, model in models.items():
                preds[name].append(R.predict([(model, "both")], sample, torch.device("cuda")))
        for name, pp in preds.items():
            overall = C.metrics(pp)
            hit = lambda p: abs(p["entry_frame"] - p["entry_gt"]) / fps[p["sample_id"]] <= .300001
            scores[name][f"k{stride}"] = {
                "n": len(pp), "score": overall["score"], "entry_acc": overall["entry_acc"],
                "collision_acc": overall["collision_acc"],
                "source_entry": {src: float(np.mean([hit(p) for p in pp if C.source(p) == src]))
                                 for src in ("AIHUB", "CCD", "MMAU", "NEXAR")},
            }
    target = root / "stage2/robust_goal_20260927/source_invariant/fold0_robust.json"
    target.write_text(json.dumps(scores, indent=2) + "\n")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
