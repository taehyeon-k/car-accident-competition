"""Night campaign — generalised distillation targets: any teacher ensemble (per fold, fold-f members on the fold's TRAINING clips, on
the student's own views k1 / k2 / k3), optional fusion of the pre-collision expert PCE into ENTRY, plus attribute targets.

Output <out>/fold{f}/k{k}/<sid>.npy float16 [2, T] (ENTRY, COLLISION) and <sid>.attr.npy float16 [2] (P(side = RIGHT), P(evasion)).
Teacher members: '<run>:<seeds>' items (seeds '012'); members with an object/actor branch get their cache from their config
(training cache = GT-COLLISION actor). ENTRY = (1 - a) mean(members) + a PCE(prefix before GT COLLISION).
Usage: python -m stage2.night.kd2 OUT_DIR --teachers E4_sa:012 E2_sa:012 XN4_sa:012 D5H:012 --pce 0.25
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from stage2.long_context_v2_experiments import common as C
from stage2.aux_signal_experiments.model import load as load_model
from stage2.generalization.robust_eval import item

RUNS = C.REPO / "stage2/generalization/results"; FOLDS = C.REPO / "stage2/long_context_v2_experiments/folds"


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--teachers", nargs="+", required=True)
    ap.add_argument("--pce", type=float, default=0.0); a = ap.parse_args(); dev = torch.device("cuda"); OUT = C.REPO / a.out
    for f in range(5):
        ms = []
        for t in a.teachers:
            run, seeds = t.split(":")
            for s in seeds:
                ck = RUNS / f"{run}/cv/fold{f}_seed{s}/checkpoint.pt"; cfg = torch.load(ck, map_location="cpu", weights_only=False)["config"]
                ms.append((load_model(ck, dev), cfg.get("obj_cache") if cfg.get("objmotion") else None))
        pce = load_model(RUNS / f"PCE/cv/fold{f}_seed0/checkpoint.pt", dev) if a.pce else None
        for k in (1, 2, 3):
            (OUT / f"fold{f}" / f"k{k}").mkdir(parents=True, exist_ok=True)
            for r in C.rows(str(FOLDS / f"fold{f}_train.jsonl")):
                its = {}; pe, pc, ps, pv = [], [], [], []
                for m, oc in ms:
                    if oc not in its: its[oc] = item(r, k, oc=oc or "stage2/actor/cache_actorfeat/a147/train")
                    it = its[oc]; T = len(it["frames"]); v = torch.ones(1, T, dtype=torch.bool, device=dev)
                    o = m(it["x"][None].to(dev), v, motion=(it["both_objfeat"] if oc else it["both"])[None].to(dev))
                    pe.append(o["entry_logits"][0].float().softmax(-1)); pc.append(o["collision_logits"][0].float().softmax(-1))
                    ps.append(o["side_logits"][0].float().softmax(-1)[1]); pv.append(o["evasion_logits"].float().sigmoid().reshape(-1)[0])
                pe, pc = torch.stack(pe).mean(0), torch.stack(pc).mean(0)
                if pce is not None:
                    it = item(r, k, oc="stage2/actor/cache_actorfeat/a147/train"); T = len(it["frames"]); x = it["x"][None].to(dev)
                    v = torch.ones(1, T, dtype=torch.bool, device=dev); mo = it["both_objfeat"][None].to(dev)
                    tc = max(int((np.asarray(it["frames"]) < int(r["collision_frame"])).sum()), min(4, T))
                    pp = torch.zeros(T, device=dev); pp[:tc] = pce(x[:, :tc], v[:, :tc], motion=mo[:, :tc])["entry_logits"][0].float().softmax(-1)
                    pe = (1 - a.pce) * pe + a.pce * pp
                sid = r["sample_id"]
                np.save(OUT / f"fold{f}" / f"k{k}" / f"{sid}.npy", torch.stack([pe, pc]).cpu().numpy().astype(np.float16))
                np.save(OUT / f"fold{f}" / f"k{k}" / f"{sid}.attr.npy", np.array([float(torch.stack(ps).mean()), float(torch.stack(pv).mean())], np.float16))
        print("fold", f, "done", flush=True)


if __name__ == "__main__": main()
