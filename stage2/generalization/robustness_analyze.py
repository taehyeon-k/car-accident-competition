"""Analyse results/robustness_profile.json: slice robustness of each recipe along frame rate, duration, frame count, ENTRY->COLLISION
gap, event position and source; clip-level drivers of misses; and which recipe-level robustness features track the leaderboard.
All metrics are on duplicate-clean clips (same exclusion as clean_eval.py). Usage:
python -m stage2.generalization.robustness_analyze [--profile results/robustness_profile.json] [--out results/robustness_analysis.json]
"""
from __future__ import annotations

import argparse
import itertools
import json
import math

import numpy as np

from stage2.long_context_v2_experiments import common as C

LB = {"v5": 0.5314, "v6": 0.5277, "v7": 0.5464, "v8": 0.5929, "v10": 0.5590, "v12": 0.5635}
RES = C.REPO / "stage2/generalization/results"
SRCS = ("AIHUB", "CCD", "MMAU", "NEXAR")


def excluded():
    d = json.loads((RES / "dup_audit.json").read_text())
    return {t[0] for t in d["lab_vs_lab"] if t[2] >= 0.5 and t[4] != t[5]} | {t[1] for t in d["unl_vs_lab"] if t[2] >= 0.5}


def clip_table():
    t = {}
    for r in C.rows("all"):
        fps, n = float(r["native_fps"]), int(r["num_frames"]); e, c = int(r["entry_frame"]), int(r["collision_frame"])
        t[r["sample_id"]] = dict(source=r["source_id"].split(":")[0], fps=fps, n=n, dur=n / fps, gap=(c - e) / fps, gap_frames=c - e,
                                 entry_rel=e / max(n - 1, 1), pre_entry=e / fps, post_coll=(n - 1 - c) / fps)
    return t


def bins(x, edges, labels):
    for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
        if lo <= x < hi: return lab
    return labels[-1]


AXES = {  # name: (feature fn(clip, pred, k), edges, labels)
    "effective_fps": (lambda c, p, k: c["fps"] / k, [0, 5.5, 8.5, 12.5, 20, 99], ["<=5", "6-8", "10-12", "15", "30"]),
    "duration_s": (lambda c, p, k: c["dur"], [0, 4, 8, 20, 35, 1e9], ["<4", "4-8", "8-20", "20-35", ">35"]),
    "retained_frames": (lambda c, p, k: p["n_retained"], [0, 25, 50, 128, 320, 1e9], ["<25", "25-49", "50-127", "128-319", ">=320"]),
    "gap_s": (lambda c, p, k: c["gap"], [-1, 0.5, 1.0, 1.5, 2.5, 1e9], ["<0.5", "0.5-1", "1-1.5", "1.5-2.5", ">2.5"]),
    "gap_retained_frames": (lambda c, p, k: c["gap_frames"] / k, [-1, 3, 6, 12, 25, 1e9], ["<3", "3-5", "6-11", "12-24", ">=25"]),
    "entry_rel_pos": (lambda c, p, k: c["entry_rel"], [-1, 0.25, 0.5, 0.75, 9], ["<.25", ".25-.5", ".5-.75", ">.75"]),
    "pre_entry_s": (lambda c, p, k: c["pre_entry"], [-1, 1, 2, 4, 1e9], ["<1", "1-2", "2-4", ">4"]),
    "source": (lambda c, p, k: c["source"], None, SRCS),
}


def hits(preds, ct):
    e = np.array([C.hit(p, "entry") for p in preds], float); c = np.array([C.hit(p, "collision") for p in preds], float)
    return e, c


def slice_metrics(preds):
    m = C.metrics(preds)
    return {k: m[k] for k in ("n", "score", "entry_acc", "collision_acc", "side_f1", "evasion_f1", "entry_catastrophic")} if preds else {"n": 0}


def conds(res_rec, stride_only=False):
    for key, preds in res_rec.items():
        if not key.startswith("k") or "crop" not in key: continue  # e.g. the DACON-like view (scored separately)
        k = int(key.split("_")[0][1:]); crop = float(key.split("crop")[1])
        if stride_only and crop: continue
        yield key, k, crop, preds


def slices(res_rec, ct, excl):
    """Axis slices pooled over the frame-rate conditions (k = 1..4, full clips); each clip counted once per condition."""
    out = {}
    for axis, (fn, edges, labels) in AXES.items():
        groups = {lab: [] for lab in labels}
        for key, k, crop, preds in conds(res_rec, stride_only=True):
            for p in preds:
                if p["sample_id"] in excl: continue
                c = ct[p["sample_id"]]; v = fn(c, p, k)
                lab = v if edges is None else bins(v, edges, labels)
                if lab in groups: groups[lab].append(p)
        out[axis] = {lab: slice_metrics(g) for lab, g in groups.items()}
    return out


def logit_fit(X, y, l2=1.0, iters=200):
    """L2 logistic regression by Newton steps on standardized X; returns coefficients (per SD) and in-sample AUC."""
    mu, sd = X.mean(0), X.std(0) + 1e-9; Z = np.c_[np.ones(len(X)), (X - mu) / sd]; w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w)); g = Z.T @ (p - y) + l2 * np.r_[0, w[1:]]
        H = (Z * (p * (1 - p))[:, None]).T @ Z + l2 * np.diag(np.r_[0, np.ones(len(w) - 1)])
        step = np.linalg.solve(H, g); w -= step
        if np.abs(step).max() < 1e-8: break
    s = Z @ w; order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(len(s))
    pos = y == 1; auc = (ranks[pos].mean() - (pos.sum() - 1) / 2) / max((~pos).sum(), 1) if pos.any() and (~pos).any() else float("nan")
    return w[1:], float(auc)


# retained frames = duration x effective fps exactly, and entry position = pre-entry / duration, so only separable traits enter the fit
FEATS = ["log_eff_fps", "log_duration_s", "log_gap_s", "log_pre_entry_s", "log_post_collision_s"]


def clip_drivers(res_rec, ct, excl):
    rows = []
    for key, k, crop, preds in conds(res_rec, stride_only=True):
        for p in preds:
            if p["sample_id"] in excl: continue
            c = ct[p["sample_id"]]
            rows.append(([math.log(c["fps"] / k), math.log(c["dur"]), math.log(max(c["gap"], 0.05)), math.log(max(c["pre_entry"], 0.05)),
                          math.log(max(c["post_coll"], 0.05))], C.hit(p, "entry"), C.hit(p, "collision"), c["source"]))
    X = np.array([r[0] for r in rows]); out = {}
    for ev, j in (("entry", 1), ("collision", 2)):
        y = np.array([r[j] for r in rows], float)
        w, auc = logit_fit(X, y); out[ev] = {"all": {"coef_per_sd": dict(zip(FEATS, map(float, w))), "auc": auc, "n": len(y)}}
        for s in SRCS:  # within-source fits: removes source identity (labels, appearance) as a confounder
            m = np.array([r[3] == s for r in rows]); Xs = X[m]; keep = Xs.std(0) > 1e-6
            ws, aucs = logit_fit(Xs[:, keep], y[m]); coef = dict.fromkeys(FEATS); coef.update(zip([f for f, kk in zip(FEATS, keep) if kk], map(float, ws)))
            out[ev][s] = {"coef_per_sd": coef, "auc": aucs, "n": int(m.sum())}
    return out


def ten_fps_view(res_rec, excl):
    """Each clip at the stride nearest 10 fps: CCD k1 (10), AIHUB mean of k1 (15) / k2 (7.5), MMAU / NEXAR k3 (10)."""
    by = {key: {p["sample_id"]: p for p in preds if p["sample_id"] not in excl} for key, preds in res_rec.items()}
    parts = []
    for sid, p in by["k1_crop0.0"].items():
        s = C.source(p)
        if s == "CCD": parts.append(p)
        elif s in ("MMAU", "NEXAR", "CAUSALCRASH"): parts.append(by["k3_crop0.0"][sid])
        elif s == "AIHUB": parts += [p, by["k2_crop0.0"][sid]]
    return parts


def recipe_features(res_rec, ct, excl, sl):
    f = {}; get = lambda key: [p for p in res_rec[key] if p["sample_id"] not in excl]
    for key in res_rec:
        m = C.metrics(get(key)); f[f"score@{key}"] = m["score"]; f[f"entry@{key}"] = m["entry_acc"]; f[f"coll@{key}"] = m["collision_acc"]
        if key in ("k1_crop0.0", "k3_crop0.0"): f[f"side@{key}"] = m["side_f1"]; f[f"eva@{key}"] = m["evasion_f1"]
    f["drop_native_to_k3"] = f["score@k1_crop0.0"] - f["score@k3_crop0.0"]
    f["drop_native_to_crop0.25"] = f["score@k1_crop0.0"] - f["score@k1_crop0.25"]
    f["mean_over_rates_k1-4"] = np.mean([f[f"score@k{k}_crop0.0"] for k in (1, 2, 3, 4)])
    f["worst_condition"] = min(v for k, v in f.items() if k.startswith("score@"))
    f["10fps_view"] = C.metrics(ten_fps_view(res_rec, excl))["score"]
    for s in SRCS:
        for key in ("k1_crop0.0", "k3_crop0.0"):
            f[f"{s}@{key}"] = C.metrics([p for p in get(key) if C.source(p) == s])["score"]
    for key in ("k1_crop0.0", "k3_crop0.0"): f[f"worst_source@{key}"] = min(f[f"{s}@{key}"] for s in SRCS)
    for axis, lab_d in sl.items():  # slice ENTRY / COLLISION hit rates pooled over rates
        for lab, m in lab_d.items():
            if m.get("n", 0) >= 60: f[f"entryhit[{axis}={lab}]"] = m["entry_acc"]; f[f"collhit[{axis}={lab}]"] = m["collision_acc"]
    # sensitivity slopes: least-squares slope of ENTRY / COLLISION hit vs log(effective fps) and vs log(gap), pooled over rates
    xs, gs, eh, ch = [], [], [], []
    for key, k, crop, preds in conds(res_rec, stride_only=True):
        for p in preds:
            if p["sample_id"] in excl: continue
            c = ct[p["sample_id"]]; xs.append(math.log(c["fps"] / k)); gs.append(math.log(max(c["gap"], .05)))
            eh.append(C.hit(p, "entry")); ch.append(C.hit(p, "collision"))
    xs, gs, eh, ch = map(np.array, (xs, gs, eh, ch))
    f["slope_entry_vs_log_fps"] = float(np.polyfit(xs, eh, 1)[0]); f["slope_coll_vs_log_fps"] = float(np.polyfit(xs, ch, 1)[0])
    f["slope_entry_vs_log_gap"] = float(np.polyfit(gs, eh, 1)[0])
    early = [(p["entry_frame"] - p["entry_gt"]) / ct[p["sample_id"]]["fps"] for p in get("k3_crop0.0")]
    f["entry_early_rate@k3"] = float(np.mean([e < -0.3 for e in early])); f["entry_late_rate@k3"] = float(np.mean([e > 0.3 for e in early]))
    return {k: float(v) for k, v in f.items()}


def rankdata(a):
    a = np.asarray(a); r = np.empty(len(a)); r[np.argsort(a, kind="stable")] = np.arange(len(a)); return r


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() < 1e-12: return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def lb_correlations(feats):
    names = [r for r in LB if r in feats]; y = np.array([LB[r] for r in names]); perms = list(itertools.permutations(range(len(y))))
    out = {}
    for key in feats[names[0]]:
        x = np.array([feats[r][key] for r in names])
        if np.isnan(x).any() or x.std() < 1e-12: continue
        pr, sr = pearson(x, y), pearson(rankdata(x), rankdata(y))
        loo = [pearson(np.delete(x, i), np.delete(y, i)) for i in range(len(y))]
        p_perm = float(np.mean([abs(pearson(x, y[list(p)])) >= abs(pr) - 1e-12 for p in perms]))  # exact permutation p (5! orderings)
        slope, icpt = np.polyfit(x, y, 1)
        out[key] = {"pearson": pr, "spearman": sr, "loo_min_pearson": float(min(loo)), "loo_max_pearson": float(max(loo)),
                    "perm_p": p_perm, "rank_correct": bool(sr > 0.999), "slope": float(slope), "intercept": float(icpt),
                    "values": dict(zip(names, map(float, x)))}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--profile", default=str(RES / "robustness_profile.json"))
    ap.add_argument("--out", default=str(RES / "robustness_analysis.json")); a = ap.parse_args()
    prof = json.loads(open(a.profile).read()); ct = clip_table(); excl = excluded()
    out = {"n_excluded": len(excl), "recipes": {}}
    for rec, res_rec in prof.items():
        sl = slices(res_rec, ct, excl)
        out["recipes"][rec] = {"conditions": {k: slice_metrics([p for p in v if p["sample_id"] not in excl]) for k, v in res_rec.items()},
                               "slices": sl, "drivers": clip_drivers(res_rec, ct, excl), "features": recipe_features(res_rec, ct, excl, sl)}
    feats = {r: v["features"] for r, v in out["recipes"].items()}
    out["lb_correlation"] = lb_correlations(feats)
    nat = [r for r in feats]; base = np.array([feats[r]["score@k1_crop0.0"] for r in nat])
    out["corr_with_native_cv_all_recipes"] = {k: pearson([feats[r][k] for r in nat], base) for k in feats[nat[0]]}
    C.dump(a.out, out)
    lc = out["lb_correlation"]; print(f"{len(lc)} recipe-level features tested against {len(LB)} LB points")
    for k, v in sorted(lc.items(), key=lambda kv: -abs(kv[1]["pearson"]))[:25]:
        print(f"{k:45s} r={v['pearson']:+.3f} rho={v['spearman']:+.2f} LOO[{v['loo_min_pearson']:+.2f},{v['loo_max_pearson']:+.2f}] p_perm={v['perm_p']:.3f}")


if __name__ == "__main__": main()
