"""Compact v14 search, step 1 — per-family CV probability cache for ensemble / post-processing search (no retraining).

For every family (5-fold CV checkpoints, seeds 0-1 as robustness_profile, so the LB forecast stays comparable) and every robustness
condition (robustness_profile.CONDITIONS + the DACON-like view), store per clip the seed-averaged ENTRY / COLLISION distributions over
the sampled positions, side / evasion probabilities, and the same for a head-level multi-view TTA: the model is re-run on thinned views
of the SAME sampled positions (positions[o::s], s in 2, 3, every offset; motion re-segmented over the view as the runtime would do) and
each view's distribution is spread back onto the full positions (mass of view position j shared by the positions it covers).
No new backbone passes are needed for the TTA at inference. FPS is read only by the metric (not here).
Output results/ens_cache/<family>.pkl: {condition: [per clip dict]}.
"""
from __future__ import annotations

import argparse, pickle, zlib

import numpy as np
import torch

from stage2.long_context_v2_experiments import common as C
from stage2.aux_signal_experiments.train import segments
from stage2.aux_signal_experiments.nexar_labels import motion28
from .robust_eval import STRIDE, FOLDS
from .robustness_profile import CONDITIONS, DACON_STRIDE, DACON_LEN, load_member

OUT = C.REPO / "stage2/generalization/results/ens_cache"
VIEWS = [(2, 0), (2, 1), (3, 0), (3, 1), (3, 2)]


def base(row, k, crop, dacon=False):
    """robust_eval.item / robustness_profile.item_dacon, returning also the per-retained-frame motion for view re-segmentation."""
    sid = row["sample_id"]; frames_all, feats = C.dense(sid)
    if dacon: k = DACON_STRIDE.get(row["source_id"].split(":")[0], 3)
    if k == 1:
        kept = np.arange(len(frames_all)); mot = np.load(C.DENSE / f"{sid}.motion.npy")
        res = np.load(C.REPO / f"stage2/aux_signal_experiments/cache_residual/{sid}.residual.npy")
    else:
        d = np.load(STRIDE / f"k{k}" / f"{sid}.npz"); kept, mot, res = d["kept"], d["motion"], d["residual"]
    fr_k = frames_all[kept]; n = len(fr_k)
    L = (DACON_LEN if n > DACON_LEN else None) if dacon else (max(int(round(crop * n)), 8) if crop else None)
    if L is not None:
        e = int(np.abs(fr_k - int(row["entry_frame"])).argmin()); c = int(np.abs(fr_k - int(row["collision_frame"])).argmin())
        L = max(L, c - e + 1); rng = np.random.default_rng(zlib.crc32(sid.encode())); lo, hi = max(0, c - L + 1), min(e, n - L)
        start = int(rng.integers(lo, hi + 1)) if hi >= lo else max(0, min(e, n - L))
        sl = slice(start, start + L); kept, mot, res = kept[sl], mot[sl].copy(), res[sl].copy(); mot[0] = 0; res[0] = 0
    reduced = frames_all[kept]; pos = C.select_adaptive(reduced)
    return {"sid": sid, "frames": reduced[pos], "x": torch.from_numpy(np.ascontiguousarray(feats[kept[pos]])), "pos": pos,
            "m28": motion28(mot), "res": res, "row": row, "n_native": len(frames_all)}


def inputs(it, sub):
    g = segments(it["m28"], it["pos"][sub]); r = segments(it["res"], it["pos"][sub])
    return {"global": torch.from_numpy(g), "both": torch.from_numpy(np.concatenate([g, r], 1))}


def spread(p, T, s, o):
    """view distribution over positions o, o+s, ... -> full positions: position t takes the mass of the view position whose
    segment (previous view position, this view position] contains t, shared equally."""
    full = np.zeros(T, np.float32); idx = np.arange(o, T, s)
    lo = 0
    for j, a in enumerate(idx):
        hi = a if j < len(idx) - 1 else T - 1
        full[lo:hi + 1] += p[j] / (hi - lo + 1); lo = hi + 1
    return full / full.sum()


@torch.inference_mode()
def run(models, it, dev):
    T = len(it["frames"]); out = {}
    views = [(1, 0)] + [(s, o) for s, o in VIEWS if T >= 3 * s]
    for s, o in views:
        sub = np.arange(o, T, s); x = it["x"][sub][None].to(dev); v = torch.ones(1, len(sub), dtype=torch.bool, device=dev); inp = inputs(it, sub)
        pe, pc, ps, pv = [], [], [], []
        for m, kind, _ in models:
            o_ = m(x, v, motion=inp[kind][None].to(dev)) if kind else m(x, v)
            pe.append(o_["entry_logits"].float().softmax(-1)[0]); pc.append(o_["collision_logits"].float().softmax(-1)[0])
            ps.append(o_["side_logits"].float().softmax(-1)[0, 1]); pv.append(o_["evasion_logits"].float().sigmoid().reshape(-1)[0])
        pe, pc = torch.stack(pe).mean(0).cpu().numpy(), torch.stack(pc).mean(0).cpu().numpy()
        if s > 1: pe, pc = spread(pe, T, s, o), spread(pc, T, s, o)
        out[(s, o)] = (pe, pc, float(torch.stack(ps).mean()), float(torch.stack(pv).mean()))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("families", nargs="+"); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    a = ap.parse_args(); dev = torch.device("cuda"); OUT.mkdir(parents=True, exist_ok=True)
    fams = [f for f in a.families if not (OUT / f"{f}.pkl").exists()]
    conds = [(f"k{k}_crop{c}", k, c, False) for k, c in CONDITIONS] + [("dacon10fps_len50", 1, 0.0, True)]
    res = {f: {c[0]: [] for c in conds} for f in fams}
    for fold in range(5):
        mem = {f: [load_member(f, fold, s, dev) for s in a.seeds] for f in fams}
        for r in C.rows(str(FOLDS / f"fold{fold}_val.jsonl")):
            for name, k, c, dac in conds:
                it = base(r, k, c, dac)
                for f in fams:
                    o = run(mem[f], it, dev)
                    res[f][name].append({"sid": it["sid"], "frames": it["frames"], "row": {k_: r[k_] for k_ in ("source_id", "entry_frame", "collision_frame", "entry_side", "evasion_space")}, "n_native": it["n_native"], "views": o})
        del mem; torch.cuda.empty_cache(); print("fold", fold, "done", flush=True)
    for f in fams: pickle.dump(res[f], open(OUT / f"{f}.pkl", "wb"))


if __name__ == "__main__": main()
