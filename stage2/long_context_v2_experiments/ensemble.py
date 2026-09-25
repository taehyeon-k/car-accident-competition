"""Probability ensemble of all seeds of a run (fixed split) and of all CV seeds per fold (pooled OOF)."""
from __future__ import annotations

import argparse, json

import numpy as np
import torch

from . import common as C
from .models import LCPyramid

R = C.REPO / "stage2/long_context_v2_experiments/results"


def load(path, device):
    st = torch.load(path, map_location="cpu", weights_only=False); cfg = st["config"]
    m = LCPyramid(pool=cfg.get("pool", "avg"), phase=3 if cfg.get("phase_weight") else 0, coarse=cfg.get("coarse", "none"))
    m.load_state_dict(st["model"]); return m.to(device).eval()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); a = ap.parse_args()
    dev = torch.device("cuda"); torch.set_num_threads(4); out = {}
    for run in a.runs:
        ck = sorted((R / run).glob("seed*/checkpoint.pt"))
        ens = C.ProbEnsemble([load(p, dev) for p in ck]).eval()
        fixed = C.breakdown(C.run(ens, [C.make_item(r, "adaptive") for r in C.rows("val")], dev))
        oof = []
        for k in range(5):
            cks = sorted((R / run / "cv").glob(f"fold{k}_seed*/checkpoint.pt"))
            e = C.ProbEnsemble([load(p, dev) for p in cks]).eval()
            vr = C.rows(str(C.REPO / f"stage2/long_context_v2_experiments/folds/fold{k}_val.jsonl"))
            oof += C.run(e, [C.make_item(r, "adaptive") for r in vr], dev)
        cv = C.breakdown(oof)
        n_cv = len(sorted((R / run / "cv").glob("fold0_seed*/checkpoint.pt")))
        out[run] = {"fixed_ensemble": fixed, "cv_ensemble": cv, "n_fixed": len(ck), "n_cv_seeds": n_cv}
        print(f"{run:16s} fixed ens({len(ck)}): {C.short_table(fixed)}\n{'':16s} CV ens({n_cv}/fold): {C.short_table(cv)}", flush=True)
    C.dump(R / f"ensemble_{'_'.join(a.runs)}.json", out)


if __name__ == "__main__": main()
