"""Locked duplicate-clean fold0 evaluation of the R3D feature pilot."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from stage2.long_context_v2_experiments import common as C
from stage2.generalization import robust_eval as R
from stage2.generalization.clean_eval import EXCL
from stage2.aux_signal_experiments.model import load as load_base
from stage2.robust_goal_20260927.r3d_head import CACHE, load as load_r3d


def main():
    device = torch.device("cuda")
    models = {
        "E4_sa": load_base(C.REPO / "stage2/generalization/results/E4_sa/cv/fold0_seed0/checkpoint.pt", device),
        "R3D_A": load_r3d(C.REPO / "stage2/robust_goal_20260927/r3d_pilot/cv/fold0_seed0/checkpoint.pt", device),
    }
    rows = [row for row in C.rows(str(R.FOLDS / "fold0_val.jsonl")) if row["sample_id"] not in EXCL]
    fps = C.fps_table()
    scores = {name: {} for name in models}
    for stride in (1, 2, 3):
        preds = {name: [] for name in models}
        for row in rows:
            item = R.item(row, stride)
            extra = np.load(CACHE / f"k{stride}" / f"{row['sample_id']}.npy")
            assert len(extra) == len(item["frames"])
            item["both_r3d"] = torch.cat((item["both"], torch.from_numpy(extra.astype(np.float32))), 1)
            for name, model in models.items():
                kind = "both" if name == "E4_sa" else "both_r3d"
                preds[name].append(R.predict([(model, kind)], item, device))
        for name, pp in preds.items():
            metric = C.metrics(pp)
            source_scores = {src: C.metrics([p for p in pp if C.source(p) == src])["score"]
                             for src in ("AIHUB", "CCD", "MMAU", "NEXAR")}
            long = [p for p in pp if (p["collision_gt"] - p["entry_gt"]) / fps[p["sample_id"]] >= 1.5]
            hit = lambda p: abs(p["entry_frame"] - p["entry_gt"]) / fps[p["sample_id"]] <= .300001
            scores[name][f"k{stride}"] = {
                "n": len(pp), "score": metric["score"], "entry_acc": metric["entry_acc"],
                "collision_acc": metric["collision_acc"], "side_f1": metric["side_f1"],
                "evasion_f1": metric["evasion_f1"], "source_score": source_scores,
                "long_gap_n": len(long), "long_gap_entry_acc": float(np.mean([hit(p) for p in long])),
            }
        root = C.REPO / "stage2/robust_goal_20260927/r3d_pilot"
        for name in models:
            (root / f"{name}_k{stride}_predictions.json").write_text(json.dumps(preds[name]) + "\n")
    root = C.REPO / "stage2/robust_goal_20260927/r3d_pilot"
    (root / "scorecard.json").write_text(json.dumps(scores, indent=2) + "\n")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
