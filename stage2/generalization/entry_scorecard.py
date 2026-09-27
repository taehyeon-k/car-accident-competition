"""ENTRY-focused robustness scorecard for CV arms (duplicate-clean clips, 3-seed probability-averaged ensembles, plain decoding).

Targets the failure modes of reports/stage2_robustness_profile.md:
  * ENTRY hit rate by ENTRY->COLLISION gap (<0.5, 0.5-1, 1-1.5, 1.5-2.5, >2.5 s), pooled over native and 1/3 frame rate
  * ENTRY hit rate by pre-ENTRY context (<=2 s vs >2 s of lead-in), pooled
  * truncation / position prior: random 25 % windows containing ENTRY..COLLISION (robust_eval.item crop) -> ENTRY hit and the slope of
    predicted vs true ENTRY position inside the window (1 = follows the evidence)
  * overall score native / 1/3 rate, ENTRY / COLLISION hit.
FPS is read only by the metric and for binning. Usage: python -m stage2.generalization.entry_scorecard ARM [ARM ...] [--seeds 0 1 2]
"""
from __future__ import annotations

import argparse, json

import numpy as np
import torch

from stage2.long_context_v2_experiments import common as C
from stage2.aux_signal_experiments.model import load as load_model
from .robust_eval import item, predict, root, FOLDS
from .clean_eval import EXCL

GAPS = [(0, .5), (.5, 1), (1, 1.5), (1.5, 2.5), (2.5, 99)]


def members(run, fold, seeds, dev):
    out = []
    for r in run.split("+"):
        for s in seeds:
            c = root(r) / r / "cv" / f"fold{fold}_seed{s}" / "checkpoint.pt"
            if not c.exists(): continue
            cfg = torch.load(c, map_location="cpu", weights_only=False)["config"]
            kind = {"both": "both", "global": "global"}.get(cfg.get("motion", "none"), None)
            if cfg.get("objmotion"): kind = "both_objfeat" if cfg.get("obj_cache") else "both_obj"
            out.append((load_model(c, dev), kind, cfg.get("feats_dir", "")))
    return out


ALPHA = None  # D: ENTRY = earliest position <= COLLISION whose probability >= ALPHA x the max there (None = joint argmax decoding)


LAMBDA = 0.0  # gap-prior decoding weight (models with a gap head): score(e, c) = log pe + log pc + LAMBDA * log N(log(1+c-e); mu, sigma)


def predict_gap(ms, it, dev, p):
    v = torch.ones(1, len(it["frames"]), dtype=torch.bool, device=dev); x = it["x"][None].to(dev); pe, pc, mu, ls = [], [], [], []
    for m, kind, *fd in ms:
        xm = x if not (fd and fd[0]) else torch.from_numpy(np.ascontiguousarray(np.load(f"{fd[0]}/{it['sid']}.npy", mmap_mode="r")[it["abs"]]))[None].to(dev)
        o = m(xm, v, motion=it[kind][None].to(dev)) if kind else m(xm, v)
        pe.append(o["entry_logits"].float().softmax(-1)); pc.append(o["collision_logits"].float().softmax(-1))
        if "gap_mu" in o: mu.append(float(o["gap_mu"][0])); ls.append(float(o["gap_logsig"][0]))
    if not mu: return p
    le = torch.stack(pe).mean(0)[0].clamp_min(1e-12).log().cpu().numpy(); lc = torch.stack(pc).mean(0)[0].clamp_min(1e-12).log().cpu().numpy()
    T = len(le); d = np.arange(T)[None, :] - np.arange(T)[:, None]  # c - e
    sig = float(np.exp(np.mean(ls))); m_ = float(np.mean(mu))
    prior = -0.5 * ((np.log1p(np.clip(d, 0, None)) - m_) / sig) ** 2
    sc = le[:, None] + lc[None, :] + LAMBDA * prior; sc[d < 0] = -np.inf
    e, c = np.unravel_index(int(np.argmax(sc)), sc.shape); fr = it["frames"]
    p["entry_frame"], p["collision_frame"] = int(fr[e]), int(fr[c]); return p


def predict_alpha(ms, it, dev):
    p = predict(ms, it, dev)
    if LAMBDA: return predict_gap(ms, it, dev, p)
    if ALPHA is None: return p
    v = torch.ones(1, len(it["frames"]), dtype=torch.bool, device=dev); x = it["x"][None].to(dev); pes = []
    for m, kind, *fd in ms:
        xm = x if not (fd and fd[0]) else torch.from_numpy(np.ascontiguousarray(np.load(f"{fd[0]}/{it['sid']}.npy", mmap_mode="r")[it["abs"]]))[None].to(dev)
        o = m(xm, v, motion=it[kind][None].to(dev)) if kind else m(xm, v); pes.append(o["entry_logits"].float().softmax(-1))
    pe = torch.stack(pes).mean(0)[0].cpu().numpy(); fr = it["frames"]
    ci = int(np.abs(fr - p["collision_frame"]).argmin()); w = pe[:ci + 1]
    ei = int(np.nonzero(w >= ALPHA * w.max())[0][0]); p["entry_frame"] = int(fr[ei])
    return p


@torch.inference_mode()
def score(run, seeds, dev):
    fps = C.fps_table(); P = {"k1": [], "k3": [], "crop": []}
    for f in range(5):
        ms = members(run, f, seeds, dev)
        for r in C.rows(str(FOLDS / f"fold{f}_val.jsonl")):
            if r["sample_id"] in EXCL: continue
            for key, (k, cr) in {"k1": (1, 0.0), "k3": (3, 0.0), "crop": (1, 0.25)}.items():
                it = item(r, k, cr); p = predict_alpha(ms, it, dev)
                p["_fps"] = fps[r["sample_id"]]; p["_lo"] = int(it["frames"][0]); p["_hi"] = int(it["frames"][-1])
                P[key].append(p)
    hit = lambda p, e: abs(p[f"{e}_frame"] - p[f"{e}_gt"]) / p["_fps"] <= .300001
    out = {}
    for key in ("k1", "k3"):
        m = C.metrics(P[key]); out[key] = {x: round(m[x], 4) for x in ("score", "entry_acc", "collision_acc", "side_f1", "evasion_f1")}
    pooled = P["k1"] + P["k3"]
    gap = lambda p: (p["collision_gt"] - p["entry_gt"]) / p["_fps"]
    out["entry_by_gap"] = {f"{lo}-{hi}": (round(float(np.mean([hit(p, "entry") for p in pooled if lo <= gap(p) < hi])), 3),
                                          sum(lo <= gap(p) < hi for p in pooled)) for lo, hi in GAPS}
    out["coll_by_gap"] = {f"{lo}-{hi}": round(float(np.mean([hit(p, "collision") for p in pooled if lo <= gap(p) < hi])), 3) for lo, hi in GAPS}
    lead = lambda p: p["entry_gt"] / p["_fps"]  # lead-in before ENTRY (clip starts at frame 0)
    out["entry_by_leadin"] = {"<=2s": round(float(np.mean([hit(p, "entry") for p in pooled if lead(p) <= 2])), 3),
                              ">2s": round(float(np.mean([hit(p, "entry") for p in pooled if lead(p) > 2])), 3)}
    cr = P["crop"]; rel = lambda x, p: (x - p["_lo"]) / max(p["_hi"] - p["_lo"], 1)
    t = np.array([rel(p["entry_gt"], p) for p in cr]); y = np.array([rel(p["entry_frame"], p) for p in cr])
    out["crop25"] = {"score": round(C.metrics(cr)["score"], 4), "entry_acc": round(float(np.mean([hit(p, "entry") for p in cr])), 3),
                     "slope": round(float(np.polyfit(t, y, 1)[0]), 3), "mean_pred_pos": round(float(y.mean()), 3), "mean_true_pos": round(float(t.mean()), 3)}
    long_gap = [p for p in pooled if gap(p) >= 1.5]
    out["entry_gap>=1.5_bias_s"] = round(float(np.median([(p["entry_frame"] - p["entry_gt"]) / p["_fps"] for p in long_gap])), 3)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default=None); ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--gap-lambda", type=float, default=0.0); a = ap.parse_args()
    ap2 = a
    global ALPHA, LAMBDA; ALPHA = a.alpha; LAMBDA = a.gap_lambda
    dev = torch.device("cuda"); res = {}
    for run in a.runs:
        res[run] = s = score(run, a.seeds, dev)
        g = " ".join(f"{k}:{v[0]:.2f}" for k, v in s["entry_by_gap"].items())
        print(f"{run:24s} S k1 {s['k1']['score']:.3f} k3 {s['k3']['score']:.3f} | E k1 {s['k1']['entry_acc']:.3f} k3 {s['k3']['entry_acc']:.3f} "
              f"| E by gap {g} | lead>2s {s['entry_by_leadin']['>2s']:.2f} | crop25 E {s['crop25']['entry_acc']:.2f} slope {s['crop25']['slope']:.2f} "
              f"| long-gap bias {s['entry_gap>=1.5_bias_s']:+.2f}s", flush=True)
    if a.out: C.dump(a.out, res)


if __name__ == "__main__": main()
