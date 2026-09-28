"""Distillation targets: teacher ENTRY / COLLISION distributions for every TRAINING clip of every CV fold, on the student's own input
views (native, 1/2, 1/3 = the stride-augmentation views of train.py; identical adaptive positions).

Teacher of fold f (no validation labels involved): mean of the fold-f D5H members (seeds 0-2; actor features from the GT-COLLISION
actor cache used in their training) for ENTRY and COLLISION, fused for ENTRY with the fold-f pre-collision expert PCE (seed 0) run on
the prefix of the same positions before GT COLLISION (zero mass after it):  ENTRY = (1 - A) D5H + A PCE, A = 0.25 (the screened fusion).
Output cache_kd/fold{f}/k{k}/{sid}.npy float16 [2, T] (ENTRY, COLLISION); --full: teachers refit on all clips -> cache_kd/all (targets
for the full-data student refit). Usage: python -m stage2.actor.kd_cache [--full]
"""
from __future__ import annotations

import numpy as np
import torch

from stage2.long_context_v2_experiments import common as C
from stage2.aux_signal_experiments.model import load as load_model
from stage2.generalization.robust_eval import item

OUT = C.REPO / "stage2/actor/cache_kd"; RUNS = C.REPO / "stage2/generalization/results"; FOLDS = C.REPO / "stage2/long_context_v2_experiments/folds"
OC = "stage2/actor/cache_actorfeat/a147/train"; A = 0.25


@torch.inference_mode()
def main():
    import sys
    dev = torch.device("cuda"); full = "--full" in sys.argv  # --full: teachers refit on all clips (D5H_full seeds 0-3, PCE_full) -> cache_kd/all
    for f in (["all"] if full else range(5)):
        if full:
            d5 = [load_model(RUNS / f"D5H_full/seed{s}/checkpoint.pt", dev) for s in range(4)]
            pce = load_model(RUNS / "PCE_full/seed0/checkpoint.pt", dev); rows = C.rows("all")
        else:
            d5 = [load_model(RUNS / f"D5H/cv/fold{f}_seed{s}/checkpoint.pt", dev) for s in range(3)]
            pce = load_model(RUNS / f"PCE/cv/fold{f}_seed0/checkpoint.pt", dev); rows = C.rows(str(FOLDS / f"fold{f}_train.jsonl"))
        sub = "all" if full else f"fold{f}"
        for k in (1, 2, 3):
            (OUT / sub / f"k{k}").mkdir(parents=True, exist_ok=True)
            for r in rows:
                it = item(r, k, oc=OC); T = len(it["frames"]); x = it["x"][None].to(dev); v = torch.ones(1, T, dtype=torch.bool, device=dev)
                mo = it["both_objfeat"][None].to(dev)
                outs = [m(x, v, motion=mo) for m in d5]
                pe = torch.stack([o["entry_logits"][0].float().softmax(-1) for o in outs]).mean(0)
                pc = torch.stack([o["collision_logits"][0].float().softmax(-1) for o in outs]).mean(0)
                tc = max(int((np.asarray(it["frames"]) < int(r["collision_frame"])).sum()), min(4, T))
                oe = pce(x[:, :tc], v[:, :tc], motion=mo[:, :tc]); pp = torch.zeros(T, device=dev); pp[:tc] = oe["entry_logits"][0].float().softmax(-1)
                np.save(OUT / sub / f"k{k}" / f"{r['sample_id']}.npy", torch.stack([(1 - A) * pe + A * pp, pc]).cpu().numpy().astype(np.float16))
        print("fold", f, "done", flush=True)


if __name__ == "__main__": main()
