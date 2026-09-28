"""Compact v14 search, step 2 — score ensembles of already-trained (packageable, no object pipeline) families + head-level TTA.

Reads results/ens_cache/<family>.pkl (ens_cache.py). A recipe = ENTRY families, COLLISION families (side / evasion over all members, as
the runtime) and a TTA weight w: p = (1 - w) * p_full + w * mean(p_views). Decoding = constrained_anchors (as robustness_profile).
LB forecast = the two-factor fit (score at 1/3 rate, crop-0.25 ENTRY accuracy; duplicate-clean clips) on the LB-scored recipes of
robustness_profile.json. Usage: python -m stage2.generalization.ens_search
"""
from __future__ import annotations

import json, pickle

import numpy as np
import torch

from stage2.long_context_v2_experiments import common as C
from stage2.spotting_experiments.objective import constrained_anchors
from .robustness_analyze import LB, excluded

CACHE = C.REPO / "stage2/generalization/results/ens_cache"
EX = excluded(); _F = {}


def fam(f):
    if f not in _F: _F[f] = pickle.load(open(CACHE / f"{f}.pkl", "rb"))
    return _F[f]


def decode(entry, coll, w=0.0, cond="k3_crop0.0"):
    fams = sorted(set(entry) | set(coll)); data = {f: fam(f)[cond] for f in fams}; preds = []
    for i, rec in enumerate(data[fams[0]]):
        if rec["sid"] in EX: continue
        def dist(f, j):
            v = data[f][i]["views"]; full = v[(1, 0)][j]; other = [x[j] for key, x in v.items() if key != (1, 0)]
            return (1 - w) * full + w * np.mean(other, 0) if (w and other) else full
        pe = np.mean([dist(f, 0) for f in entry], 0); pc = np.mean([dist(f, 1) for f in coll], 0)
        side = np.mean([data[f][i]["views"][(1, 0)][2] for f in fams]); eva = np.mean([data[f][i]["views"][(1, 0)][3] for f in fams])
        ei, ci = constrained_anchors(torch.tensor(pe)[None].clamp_min(1e-12).log(), torch.tensor(pc)[None].clamp_min(1e-12).log())
        r = rec["row"]; fr = rec["frames"]
        preds.append({"sample_id": rec["sid"], "source_id": r["source_id"], "entry_frame": int(fr[int(ei[0])]), "collision_frame": int(fr[int(ci[0])]),
                      "entry_side": int(side >= .5), "evasion_space": int(eva >= .5), "entry_gt": int(r["entry_frame"]),
                      "collision_gt": int(r["collision_frame"]), "entry_side_gt": int(r["entry_side"] == "RIGHT"),
                      "evasion_gt": int(r["evasion_space"]), "num_available_frames": rec["n_native"]})
    return preds


def forecast_model():
    res = json.load(open(C.REPO / "stage2/generalization/results/robustness_profile.json"))
    g = lambda rec, k: C.metrics([p for p in res[rec][k] if p["sample_id"] not in EX])
    X = np.array([[1, g(k, "k3_crop0.0")["score"], g(k, "k1_crop0.25")["entry_acc"]] for k in LB]); y = np.array(list(LB.values()))
    return np.linalg.lstsq(X, y, rcond=None)[0]


W = None


def evaluate(entry, coll, w=0.0, full=False):
    global W
    if W is None: W = forecast_model()
    m3 = C.metrics(decode(entry, coll, w, "k3_crop0.0")); m25 = C.metrics(decode(entry, coll, w, "k1_crop0.25"))
    out = {"third": m3["score"], "crop25E": m25["entry_acc"], "forecast": float(W @ [1, m3["score"], m25["entry_acc"]])}
    if full:
        for c in ("k1_crop0.0", "k2_crop0.0", "k4_crop0.0", "k1_crop0.5", "k3_crop0.5", "dacon10fps_len50"):
            m = C.metrics(decode(entry, coll, w, c)); out[c] = m["score"]; out[c + "_E"] = m["entry_acc"]
        out["third_E"] = m3["entry_acc"]; out["crop25"] = m25["score"]
    return out
