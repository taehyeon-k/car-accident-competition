"""Test-format validation view (2026-09-29): does a view built from the DACON Stage 2 sample format rank the leaderboard?

DACON Stage 2 samples: 1280x720, 10 fps, exactly 50 frames, COLLISION at frames 30-41. View per duplicate-clean clip: the source's
~10 fps stride (CCD 1, AIHUB 2, MMAU / NEXAR 3; stride caches as robustness_profile), then a 50-frame window whose COLLISION falls at
a seeded position p ~ U{30..41} (window start = COLLISION - p, clamped to the clip); clips with <= 50 retained frames are used whole
(all CCD clips: the test format itself). ENTRY may fall outside the window when the gap is long (as it could in the test).
Recipes = robustness_profile.RECIPES (seeds 0-1, plain decoding) + v8kde (KDe + E2_sa + XN4_sa). Reports score / ENTRY / COLLISION /
side / evasion per recipe and correlation with the Stage-2-only leaderboard scores. FPS is used only by the metric.
Usage: python -m stage2.generalization.test50_view
"""
from __future__ import annotations

import json, zlib

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from stage2.long_context_v2_experiments import common as C
from stage2.aux_signal_experiments.train import segments
from stage2.aux_signal_experiments.nexar_labels import motion28
from .robust_eval import STRIDE, FOLDS
from .robustness_profile import RECIPES, load_member, predict, DACON_STRIDE
from .clean_eval import EXCL

LB = {"v5": .5314, "v6": .5277, "v7": .5464, "v8": .59293, "v10": .5590, "v12": .5635, "v13": .5609, "v8kde": .5886}
L = 50


def item_test50(row):
    sid = row["sample_id"]; k = DACON_STRIDE.get(row["source_id"].split(":")[0], 3); frames_all, feats = C.dense(sid)
    if k == 1:
        kept = np.arange(len(frames_all)); mot = np.load(C.DENSE / f"{sid}.motion.npy")
        res = np.load(C.REPO / f"stage2/aux_signal_experiments/cache_residual/{sid}.residual.npy")
    else:
        d = np.load(STRIDE / f"k{k}" / f"{sid}.npz"); kept, mot, res = d["kept"], d["motion"], d["residual"]
    fr_k = frames_all[kept]; n = len(fr_k)
    if n > L:
        c = int(np.abs(fr_k - int(row["collision_frame"])).argmin()); p = int(np.random.default_rng(zlib.crc32(sid.encode())).integers(30, 42))
        start = int(np.clip(c - p, 0, n - L)); sl = slice(start, start + L)
        kept, mot, res = kept[sl], mot[sl].copy(), res[sl].copy(); mot[0] = 0; res[0] = 0
    reduced = frames_all[kept]; pos = C.select_adaptive(reduced); frames = reduced[pos]
    g = segments(motion28(mot), pos); r = segments(res, pos); both = np.concatenate([g, r], 1)
    return {"sid": sid, "frames": frames, "x": torch.from_numpy(np.ascontiguousarray(feats[kept[pos]])), "global": torch.from_numpy(g),
            "both": torch.from_numpy(both), "abs": kept[pos], "n": len(reduced), "n_native": len(frames_all), "row": row, "both_objfeat": None}


def main():
    dev = torch.device("cuda"); fps = C.fps_table(); recipes = dict(RECIPES)
    recipes["v8kde"] = {**RECIPES["v8"], "entry": ["KDe", "E2_sa", "XN4_sa"], "coll": ["KDe", "E2_sa", "XN4_sa"], "attr": ["KDe", "E2_sa", "XN4_sa"]}
    out = {}
    for rec in LB:
        spec = recipes[rec]; fams = sorted(set(spec["entry"] + spec["coll"] + spec["attr"])); preds = []
        for f in range(5):
            members = {fam: [load_member(fam, f, s, dev) for s in (0, 1)] for fam in fams}
            for r in C.rows(str(FOLDS / f"fold{f}_val.jsonl")):
                if r["sample_id"] in EXCL: continue
                preds.append(predict(members, spec, item_test50(r), dev))
        m = C.metrics(preds); inwin = np.mean([p["entry_gt"] >= p.get("_lo", -1) for p in preds]) if preds and "_lo" in preds[0] else float("nan")
        out[rec] = {k: m[k] for k in ("score", "entry_acc", "collision_acc", "side_f1", "evasion_f1")}
        out[rec]["by_source"] = {s: C.metrics([p for p in preds if C.source(p) == s])["score"] for s in ("AIHUB", "CCD", "MMAU", "NEXAR")}
        print(rec, LB[rec], {k: round(v, 4) for k, v in out[rec].items() if k != "by_source"}, {k: round(v, 3) for k, v in out[rec]["by_source"].items()}, flush=True)
    y = np.array([LB[r] for r in LB])
    for key in ("score", "entry_acc", "collision_acc", "side_f1", "evasion_f1"):
        x = np.array([out[r][key] for r in LB]); print(f"{key:14s} pearson {pearsonr(x, y)[0]:+.2f} spearman {spearmanr(x, y)[0]:+.2f}")
    C.dump(C.REPO / "stage2/generalization/results/test50_view.json", out)


if __name__ == "__main__": main()
