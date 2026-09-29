"""Matched fold-0 robustness check for the isolated actor-scene pilot."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from stage2.long_context_v2_experiments import common as C
from stage2.generalization import robust_eval as R
from stage2.generalization.clean_eval import EXCL
from stage2.aux_signal_experiments.model import load as load_standard
from stage2.robust_goal_20260927.actor_interaction import load as load_interaction


def main():
    os.environ["OBJ_CACHE"] = "stage2/objtrack/cache_objlane_full"
    repo = C.REPO
    paths = {
        "actor_scene": repo / "stage2/robust_goal_20260927/actor_scene/cv/fold0_seed0/checkpoint.pt",
        "actor_scene_frozen": repo / "stage2/robust_goal_20260927/actor_scene_frozen/cv/fold0_seed0/checkpoint.pt",
        "OB_D": repo / "stage2/generalization/results/OB_D/cv/fold0_seed0/checkpoint.pt",
        "E4_sa": repo / "stage2/generalization/results/E4_sa/cv/fold0_seed0/checkpoint.pt",
    }
    device = torch.device("cuda")
    models = {name: (load_interaction(path, device) if name.startswith("actor_scene") else load_standard(path, device)) for name, path in paths.items()}
    rows = [r for r in C.rows(str(R.FOLDS / "fold0_val.jsonl")) if r["sample_id"] not in EXCL]
    fps = C.fps_table()
    result = {name: {} for name in models}
    for k in (1, 2, 3):
        preds = {name: [] for name in models}
        for row in rows:
            sample = R.item(row, k)
            for name, model in models.items():
                preds[name].append(R.predict([(model, "both" if name == "E4_sa" else "both_objfeat")], sample, device))
        for name, pp in preds.items():
            metric = C.metrics(pp)
            long = [p for p in pp if (p["collision_gt"] - p["entry_gt"]) / fps[p["sample_id"]] >= 1.5]
            hit = lambda p: abs(p["entry_frame"] - p["entry_gt"]) / fps[p["sample_id"]] <= .300001
            result[name][f"k{k}"] = {
                "n": len(pp), "score": metric["score"], "entry_acc": metric["entry_acc"],
                "collision_acc": metric["collision_acc"],
                "long_gap_n": len(long), "long_gap_entry_acc": float(np.mean([hit(p) for p in long])) if long else None,
                "nexar_entry_acc": float(np.mean([hit(p) for p in pp if C.source(p) == "NEXAR"])),
            }
    out = repo / "stage2/robust_goal_20260927/actor_scene/fold0_robust.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
