"""Per-clip robustness profile of the recent Stage 2 recipes (v5-v12 CV proxies).

For every recipe, the 5-fold CV checkpoints (seeds 0-1 of every family, matched across recipes) predict their held-out clips under
several input conditions, and every clip's prediction is saved, so any slice (frame rate, duration, frame count, ENTRY->COLLISION gap,
event position, source) can be scored afterwards with the official metric (robustness_analyze.py).
Conditions: frame rate 1/k (k = 1..4, native frames[::k], motion recomputed; stride caches from extract_stride_motion.py) and
duration crops (random window of crop x N retained frames that still contains ENTRY..COLLISION; robust_eval.item).
Decoding: plain (probability average + constrained anchors), as robust_eval / v56_view, so it measures the models, not the fusion.
FPS is read only by the metric. Usage: python -m stage2.generalization.robustness_profile --out results/robustness_profile.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from stage2.long_context_v2_experiments import common as C
from stage2.long_context_v2_experiments.ensemble import load as load_lc
from stage2.aux_signal_experiments.model import load as load_aux
from stage2.spotting_experiments.objective import constrained_anchors
from .robust_eval import item, FOLDS

LC = C.REPO / "stage2/long_context_v2_experiments/results"
AUX_ROOTS = [C.REPO / "stage2/aux_signal_experiments/results", C.REPO / "stage2/generalization/results"]
V5 = ["C0_avg", "X_ema", "M_motion", "PH_phase"]


def same(fams): return dict(entry=fams, coll=fams, attr=fams)


RECIPES = {  # LB-scored: v5 0.5314, v6 0.5277, v7 0.5464, v8 0.5929, v10 0.5590; v9 / v11 / v12 unscored
    "v5": same(V5),
    "v6": dict(entry=["E2_NT_both_bnd2", "M1_both", "PH_phase", "E4_NT_both", "M_motion"], coll=["E2_NT_both_bnd2"],
               attr=["C0_avg", "X_ema", "M_motion", "PH_phase", "M1_both", "E4_NT_both", "E2_NT_both_bnd2"]),
    "v7": same(["E4_NT_both", "E2_NT_both_bnd2", "XN4_both_nexar"]),
    "v8": same(["E4_sa", "E2_sa", "XN4_sa"]),
    "v9": same(["E4_sa_ema", "E2_sa_ema", "XN4_sa_ema"]),
    "v10": same(["XC_E4", "XS_E2", "XS_XN4"]),
    "v11": same(["XSbU_E4", "XSbU_E2", "XSbU_XN4"]),
    "v12": same(["XSbU_E4", "XSbU_E2", "XSbU_XN4", "XSbUB_E4"]),
    "v13": same(["E4_sbuC", "E2_sbuC", "XN4_sbuC"]),    # stride mix + unlabelled consistency + window crops, no MM-AU/CCD extras
    "v13nc": same(["E4_sbu", "E2_sbu", "XN4_sbu"]),     # same without window crops (attribution)
    "v13+OA": same(["E4_sbuC", "E2_sbuC", "XN4_sbuC", "E4_sbOA"]),              # + object-crop E4 family (no window crops)
    "v13+OAC": same(["E4_sbuC", "E2_sbuC", "XN4_sbuC", "E4_sbOAC"]),            # + object-crop E4 family with window crops
    "v13+OAC2": same(["E4_sbuC", "E2_sbuC", "XN4_sbuC", "E4_sbOAC", "E2_sbOAC"]),  # + object-crop E4 and E2 families
    "OAC+XN4": same(["E4_sbOAC", "E2_sbOAC", "XN4_sbuC"]),                      # object-crop E4/E2 replace v13's E4/E2
    "v8+OA": same(["E4_sa", "E2_sa", "XN4_sa", "E4_sbOA"]),                      # v8 + object-crop E4 family
    "v8+OAC": same(["E4_sa", "E2_sa", "XN4_sa", "E4_sbOAC"]),                    # v8 + object-crop E4 family with window crops
}
CONDITIONS = [(1, 0.0), (2, 0.0), (3, 0.0), (4, 0.0), (1, 0.5), (1, 0.25), (3, 0.5)]
# "DACON-like" view (pre-registered before its result was seen): the known test samples are CCD clips, 10 fps, 50 frames. Each clip at the
# stride nearest 10 fps (CCD 1, AIHUB 2 = 7.5 fps, MMAU / NEXAR / CausalCrash 3), cut to a 50-retained-frame window containing ENTRY..COLLISION.
DACON_STRIDE = {"CCD": 1, "AIHUB": 2}; DACON_LEN = 50


def item_dacon(row):
    import zlib
    from stage2.aux_signal_experiments.train import segments
    from stage2.aux_signal_experiments.nexar_labels import motion28
    from .robust_eval import STRIDE
    sid = row["sample_id"]; k = DACON_STRIDE.get(row["source_id"].split(":")[0], 3); frames_all, feats = C.dense(sid)
    if k == 1:
        kept = np.arange(len(frames_all)); mot = np.load(C.DENSE / f"{sid}.motion.npy")
        res = np.load(C.REPO / f"stage2/aux_signal_experiments/cache_residual/{sid}.residual.npy")
    else:
        d = np.load(STRIDE / f"k{k}" / f"{sid}.npz"); kept, mot, res = d["kept"], d["motion"], d["residual"]
    fr_k = frames_all[kept]; n = len(fr_k)
    if n > DACON_LEN:  # same window rule as robust_eval.item's crop, with a fixed length
        e = int(np.abs(fr_k - int(row["entry_frame"])).argmin()); c = int(np.abs(fr_k - int(row["collision_frame"])).argmin())
        L = max(DACON_LEN, c - e + 1); rng = np.random.default_rng(zlib.crc32(sid.encode())); lo, hi = max(0, c - L + 1), min(e, n - L)
        start = int(rng.integers(lo, hi + 1)) if hi >= lo else max(0, min(e, n - L))
        sl = slice(start, start + L); kept, mot, res = kept[sl], mot[sl].copy(), res[sl].copy(); mot[0] = 0; res[0] = 0
    else: sl = slice(None)
    reduced = frames_all[kept]; pos = C.select_adaptive(reduced); frames = reduced[pos]
    g = segments(motion28(mot), pos); r = segments(res, pos)
    both = np.concatenate([g, r], 1); import os
    of = C.REPO / f"{os.environ.get('OBJ_CACHE', 'stage2/objtrack/cache_objfeat')}/k{k}/{sid}.npy"
    obf = np.load(of)[sl] if of.exists() else None
    return {"sid": sid, "frames": frames, "x": torch.from_numpy(np.ascontiguousarray(feats[kept[pos]])), "global": torch.from_numpy(g),
            "both": torch.from_numpy(both), "abs": kept[pos], "n": len(reduced), "n_native": len(frames_all), "row": row,
            "both_objfeat": torch.from_numpy(np.concatenate([both, segments(obf, pos)], 1)) if obf is not None else None}


def load_member(fam, fold, seed, dev):
    if (LC / fam).is_dir():
        p = LC / fam / "cv" / f"fold{fold}_seed{seed}" / "checkpoint.pt"; cfg = torch.load(p, map_location="cpu", weights_only=False)["config"]
        return load_lc(p, dev), ("global" if cfg.get("motion") else None), ""
    p = next(r / fam for r in AUX_ROOTS if (r / fam).is_dir()) / "cv" / f"fold{fold}_seed{seed}" / "checkpoint.pt"
    cfg = torch.load(p, map_location="cpu", weights_only=False)["config"]
    kind = {"both": "both", "global": "global"}.get(cfg.get("motion", "none"))
    if cfg.get("objmotion"): kind = "both_objfeat" if cfg.get("obj_cache") else "both_obj"
    return load_aux(p, dev), kind, cfg.get("feats_dir", "")


@torch.inference_mode()
def predict(members, spec, it, dev):
    v = torch.ones(1, len(it["frames"]), dtype=torch.bool, device=dev); x = it["x"][None].to(dev); xs = {}; outs = {}
    for fam, ms in members.items():
        outs[fam] = []
        for m, kind, fd in ms:
            xm = x
            if fd:  # member trained on another backbone's features: same frames from its own cache
                if fd not in xs: xs[fd] = torch.from_numpy(np.ascontiguousarray(np.load(Path(fd) / f"{it['sid']}.npy", mmap_mode="r")[it["abs"]]))[None].to(dev)
                xm = xs[fd]
            outs[fam].append(m(xm, v, motion=it[kind][None].to(dev)) if kind else m(xm, v))
    avg = lambda key, fams, fn: torch.stack([fn(o[key].float()) for f in fams for o in outs[f]]).mean(0)
    pe = avg("entry_logits", spec["entry"], lambda z: z.softmax(-1)); pc = avg("collision_logits", spec["coll"], lambda z: z.softmax(-1))
    side = float(avg("side_logits", spec["attr"], lambda z: z.softmax(-1)[0, 1]))
    eva = float(avg("evasion_logits", spec["attr"], lambda z: z.sigmoid().reshape(-1)[0]))
    ei, ci = constrained_anchors(pe.log(), pc.log()); r = it["row"]; fr = it["frames"]
    return {"sample_id": it["sid"], "source_id": r["source_id"], "entry_frame": int(fr[int(ei[0])]), "collision_frame": int(fr[int(ci[0])]),
            "entry_side": int(side >= .5), "evasion_space": int(eva >= .5), "p_side": side, "p_eva": eva,
            "entry_conf": float(pe.max()), "coll_conf": float(pc.max()),
            "entry_gt": int(r["entry_frame"]), "collision_gt": int(r["collision_frame"]), "entry_side_gt": int(r["entry_side"] == "RIGHT"),
            "evasion_gt": int(r["evasion_space"]), "num_available_frames": it["n_native"], "n_retained": it["n"], "n_positions": len(fr)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--recipes", nargs="+", default=list(RECIPES)); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--out", default=str(C.REPO / "stage2/generalization/results/robustness_profile.json")); a = ap.parse_args()
    dev = torch.device("cuda"); out = Path(a.out)
    res = __import__("json").loads(out.read_text()) if out.exists() else {}
    for rec in a.recipes:
        spec = RECIPES[rec]; fams = sorted(set(spec["entry"] + spec["coll"] + spec["attr"]))
        if rec in res and all(f"k{k}_crop{c}" in res[rec] for k, c in CONDITIONS) and "dacon10fps_len50" in res[rec]: continue
        res[rec] = {f"k{k}_crop{c}": [] for k, c in CONDITIONS}; res[rec]["dacon10fps_len50"] = []
        for f in range(5):
            members = {fam: [load_member(fam, f, s, dev) for s in a.seeds] for fam in fams}
            for r in C.rows(str(FOLDS / f"fold{f}_val.jsonl")):
                for k, c in CONDITIONS: res[rec][f"k{k}_crop{c}"].append(predict(members, spec, item(r, k, c), dev))
                res[rec]["dacon10fps_len50"].append(predict(members, spec, item_dacon(r), dev))
            del members; torch.cuda.empty_cache()
        print(rec, " ".join(f"{key} {C.metrics(p)['score']:.4f}" for key, p in res[rec].items()), flush=True)
        C.dump(out, res)  # checkpoint after every recipe


if __name__ == "__main__": main()
