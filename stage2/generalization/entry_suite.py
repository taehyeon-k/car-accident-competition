"""Experiment 0 of stage2/Stage2_Entry_Robustness_Experiments.md — ENTRY robustness evaluation suite.

For CV arms (families joined with '+', seeds 0-2 by default; duplicate-clean validation clips) under five conditions — native, 1/2 and
1/3 frame rate (motion recomputed), and random 50 % / 25 % windows containing ENTRY..COLLISION (robust_eval.item) — every member's
probabilities are computed once; the ensemble and each single seed (all families of that seed) are decoded from them (plain decoding:
probability average + constrained anchors). Per-clip predictions are saved (results/entry_suite/<arm>.json).
Reported: score / ENTRY / COLLISION per condition; ENTRY hit and signed timing error (median, mean; s) per ENTRY->COLLISION gap bin
(pooled over the three rates); crop ENTRY hit and predicted-vs-true ENTRY position slope inside the window; source-wise ENTRY;
per-seed values (variance); statistical-prior baselines (timing only, no video evidence):
  PRIOR_POS   ENTRY at the training-median relative position of ENTRY in the clip / window
  PRIOR_GAPR  ENTRY = the reference model's COLLISION - training-median gap as a fraction of the clip / window length
FPS is used only by the metric and for gap binning. `--entry-key` decodes another ENTRY output (e.g. an ENTRY branch alone).
"""
from __future__ import annotations

import argparse, json
from collections import defaultdict

import numpy as np
import torch

from stage2.long_context_v2_experiments import common as C
from stage2.aux_signal_experiments.model import load as load_model
from stage2.spotting_experiments.objective import constrained_anchors
from .robust_eval import item, root, FOLDS
from .clean_eval import EXCL

RES = C.REPO / "stage2/generalization/results/entry_suite"
CONDS = {"k1": (1, 0.0), "k2": (2, 0.0), "k3": (3, 0.0), "crop50": (1, 0.5), "crop25": (1, 0.25)}
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
            out.append((load_model(c, dev), kind, cfg.get("feats_dir", ""), s))
    return out


@torch.inference_mode()
def member_probs(ms, it, dev, entry_key):
    v = torch.ones(1, len(it["frames"]), dtype=torch.bool, device=dev); x = it["x"][None].to(dev); xs = {}; out = []
    for m, kind, fd, s in ms:
        xm = x
        if fd:
            if fd not in xs: xs[fd] = torch.from_numpy(np.ascontiguousarray(np.load(f"{fd}/{it['sid']}.npy", mmap_mode="r")[it["abs"]]))[None].to(dev)
            xm = xs[fd]
        o = m(xm, v, motion=it[kind][None].to(dev)) if kind else m(xm, v)
        ek = entry_key if entry_key in o else "entry_logits"
        out.append((s, o[ek][0].float().softmax(-1), o["collision_logits"][0].float().softmax(-1),
                    float(o["side_logits"][0].float().softmax(-1)[1]), float(o["evasion_logits"].float().sigmoid().reshape(-1)[0])))
    return out


def decode(parts, it, r):
    pe = torch.stack([p[1] for p in parts]).mean(0); pc = torch.stack([p[2] for p in parts]).mean(0)
    ei, ci = constrained_anchors(pe.log()[None], pc.log()[None]); fr = it["frames"]
    return {"sample_id": r["sample_id"], "source_id": r["source_id"], "entry_frame": int(fr[int(ei[0])]), "collision_frame": int(fr[int(ci[0])]),
            "entry_side": int(np.mean([p[3] for p in parts]) >= .5), "evasion_space": int(np.mean([p[4] for p in parts]) >= .5),
            "entry_gt": int(r["entry_frame"]), "collision_gt": int(r["collision_frame"]), "entry_side_gt": int(r["entry_side"] == "RIGHT"),
            "evasion_gt": int(r["evasion_space"]), "num_available_frames": it["n_native"], "_lo": int(fr[0]), "_hi": int(fr[-1]), "_n": len(fr)}


def run_arm(run, seeds, dev, entry_key="entry_logits"):
    P = {c: {"ens": [], **{f"s{s}": [] for s in seeds}} for c in CONDS}
    for f in range(5):
        ms = members(run, f, seeds, dev)
        for r in C.rows(str(FOLDS / f"fold{f}_val.jsonl")):
            if r["sample_id"] in EXCL: continue
            for cond, (k, cr) in CONDS.items():
                it = item(r, k, cr); parts = member_probs(ms, it, dev, entry_key)
                P[cond]["ens"].append(decode(parts, it, r))
                for s in seeds:
                    sp = [p for p in parts if p[0] == s]
                    if sp: P[cond][f"s{s}"].append(decode(sp, it, r))
    return P


def metrics(P, fps):
    hit = lambda p, e: abs(p[f"{e}_frame"] - p[f"{e}_gt"]) / fps[p["sample_id"]] <= .300001
    gap = lambda p: (p["collision_gt"] - p["entry_gt"]) / fps[p["sample_id"]]
    out = {}
    for c in CONDS:
        m = C.metrics(P[c]); out[c] = {k: round(m[k], 4) for k in ("score", "entry_acc", "collision_acc", "side_f1", "evasion_f1")}
    pooled = P["k1"] + P["k2"] + P["k3"]
    out["gap"] = {}
    for lo, hi in GAPS:
        ps = [p for p in pooled if lo <= gap(p) < hi]; err = np.array([(p["entry_frame"] - p["entry_gt"]) / fps[p["sample_id"]] for p in ps])
        out["gap"][f"{lo}-{hi}"] = {"n": len(ps), "entry_acc": round(float(np.mean([hit(p, "entry") for p in ps])), 3),
                                    "coll_acc": round(float(np.mean([hit(p, "collision") for p in ps])), 3),
                                    "bias_median_s": round(float(np.median(err)), 3), "bias_mean_s": round(float(err.mean()), 3)}
    lg = [p for p in pooled if gap(p) >= 1.5]
    out["long_gap_bias_median_s"] = round(float(np.median([(p["entry_frame"] - p["entry_gt"]) / fps[p["sample_id"]] for p in lg])), 3)
    gl = np.array([np.log(max(gap(p), .05)) for p in pooled]); hl = np.array([hit(p, "entry") for p in pooled], float)
    out["entry_vs_log_gap_slope"] = round(float(np.polyfit(gl, hl, 1)[0]), 3)
    for c in ("crop50", "crop25"):
        rel = lambda x, p: (x - p["_lo"]) / max(p["_hi"] - p["_lo"], 1)
        t = np.array([rel(p["entry_gt"], p) for p in P[c]]); y = np.array([rel(p["entry_frame"], p) for p in P[c]])
        out[c]["slope"] = round(float(np.polyfit(t, y, 1)[0]), 3); out[c]["mean_pred_pos"] = round(float(y.mean()), 3); out[c]["mean_true_pos"] = round(float(t.mean()), 3)
    out["source"] = {s: {c: round(float(np.mean([hit(p, "entry") for p in P[c] if C.source(p) == s])), 3) for c in ("k1", "k3", "crop25")}
                     for s in ("AIHUB", "CCD", "MMAU", "NEXAR")}
    return out


def prior_baselines(P_ref, fps):
    """timing-only baselines fitted per fold on the training labels (median ENTRY relative position, median gap fraction);
    COLLISION taken from the reference arm's ensemble prediction under the same condition."""
    fold_of = {}
    for f in range(5):
        for r in C.rows(str(FOLDS / f"fold{f}_val.jsonl")): fold_of[r["sample_id"]] = f
    stats = {}
    for f in range(5):
        tr = C.rows(str(FOLDS / f"fold{f}_train.jsonl"))
        rel = [int(r["entry_frame"]) / max(int(r["num_frames"]) - 1, 1) for r in tr]
        gfrac = [(int(r["collision_frame"]) - int(r["entry_frame"])) / max(int(r["num_frames"]) - 1, 1) for r in tr]
        stats[f] = (float(np.median(rel)), float(np.median(gfrac)))
    out = {}
    for name in ("PRIOR_POS", "PRIOR_GAPR"):
        Q = {}
        for c in CONDS:
            Q[c] = {"ens": []}
            for p in P_ref[c]["ens"]:
                rp, gf = stats[fold_of[p["sample_id"]]]; span = p["_hi"] - p["_lo"]; q = dict(p)
                e = p["_lo"] + rp * span if name == "PRIOR_POS" else p["collision_frame"] - gf * span
                q["entry_frame"] = int(round(min(max(e, p["_lo"]), q["collision_frame"]))); Q[c]["ens"].append(q)
        out[name] = metrics({c: Q[c]["ens"] for c in CONDS}, fps)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--entry-key", default="entry_logits"); ap.add_argument("--tag", default=""); ap.add_argument("--priors", action="store_true")
    a = ap.parse_args(); dev = torch.device("cuda"); fps = C.fps_table(); RES.mkdir(parents=True, exist_ok=True)
    for run in a.runs:
        key = run + (f"@{a.tag}" if a.tag else "")
        P = run_arm(run, a.seeds, dev, a.entry_key)
        res = {"ensemble": metrics({c: P[c]["ens"] for c in CONDS}, fps),
               "per_seed": {f"s{s}": metrics({c: P[c][f"s{s}"] for c in CONDS}, fps) for s in a.seeds}}
        if a.priors: res["priors"] = prior_baselines(P, fps)
        C.dump(RES / f"{key}.json", {"metrics": res, "preds": {c: P[c]["ens"] for c in CONDS}})
        e = res["ensemble"]; g = e["gap"]
        seeds_lg = [res["per_seed"][f"s{s}"]["gap"]["1.5-2.5"]["entry_acc"] for s in a.seeds]
        print(f"{key:36s} S {e['k1']['score']:.3f}/{e['k2']['score']:.3f}/{e['k3']['score']:.3f} | E {e['k1']['entry_acc']:.3f}/{e['k3']['entry_acc']:.3f} "
              f"| gapE " + "/".join(f"{g[b]['entry_acc']:.2f}" for b in g) + f" | 1.5-2.5 seeds {seeds_lg} | bias {e['long_gap_bias_median_s']:+.2f}s "
              f"| crop25 E {e['crop25']['entry_acc']:.3f} slope {e['crop25']['slope']:.2f} | crop50 E {e['crop50']['entry_acc']:.3f} slope {e['crop50']['slope']:.2f}", flush=True)
        if a.priors:
            for pn, pm in res["priors"].items():
                g = pm["gap"]; print(f"   {pn:12s} E {pm['k1']['entry_acc']:.3f}/{pm['k3']['entry_acc']:.3f} | gapE " + "/".join(f"{g[b]['entry_acc']:.2f}" for b in g)
                                     + f" | bias {pm['long_gap_bias_median_s']:+.2f}s | crop25 E {pm['crop25']['entry_acc']:.3f} slope {pm['crop25']['slope']:.2f}", flush=True)


if __name__ == "__main__": main()


@torch.inference_mode()
def gapood(run, seeds, dev):
    """gap-OOD: models trained on short/medium gaps (L split) scored on long-gap clips, and trained on medium/long gaps (S split) scored
    on short-gap clips; native and 1/3 rate; ensemble + per seed + a timing prior fitted on the same training split."""
    fps = C.fps_table(); G = C.REPO / "stage2/generalization/gapood"; out = {}
    for sp in ("L", "S"):
        ms = []
        for r_ in run.split("+"):
            for s in seeds:
                c = root(r_) / r_ / f"{sp}_seed{s}" / "checkpoint.pt"
                if not c.exists(): continue
                cfg = torch.load(c, map_location="cpu", weights_only=False)["config"]
                kind = {"both": "both", "global": "global"}.get(cfg.get("motion", "none"), None)
                if cfg.get("objmotion"): kind = "both_objfeat" if cfg.get("obj_cache") else "both_obj"
                ms.append((load_model(c, dev), kind, cfg.get("feats_dir", ""), s))
        tr = C.rows(str(G / f"{sp}_train.jsonl"))
        gfrac = float(np.median([(int(r["collision_frame"]) - int(r["entry_frame"])) / max(int(r["num_frames"]) - 1, 1) for r in tr]))
        P = {k: {"ens": [], "prior": [], **{f"s{s}": [] for s in seeds}} for k in ("k1", "k3")}
        for r in C.rows(str(G / f"{sp}_val.jsonl")):
            if r["sample_id"] in EXCL: continue
            for key, k in (("k1", 1), ("k3", 3)):
                it = item(r, k); parts = member_probs(ms, it, dev, "entry_logits"); p = decode(parts, it, r); P[key]["ens"].append(p)
                q = dict(p); q["entry_frame"] = int(round(max(p["_lo"], p["collision_frame"] - gfrac * (p["_hi"] - p["_lo"])))); P[key]["prior"].append(q)
                for s in seeds:
                    sp_ = [x for x in parts if x[0] == s]
                    if sp_: P[key][f"s{s}"].append(decode(sp_, it, r))
        hit = lambda p: abs(p["entry_frame"] - p["entry_gt"]) / fps[p["sample_id"]] <= .300001
        bias = lambda ps: float(np.median([(p["entry_frame"] - p["entry_gt"]) / fps[p["sample_id"]] for p in ps]))
        out.setdefault("_preds", {})[sp] = {key: P[key]["ens"] for key in P}
        out[sp] = {key: {m: {"entry_acc": round(float(np.mean([hit(p) for p in P[key][m]])), 3), "bias_s": round(bias(P[key][m]), 3),
                             "score": round(C.metrics(P[key][m])["score"], 4)} for m in P[key]} for key in P}
    return out


def gapood_main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); a = ap.parse_args()
    dev = torch.device("cuda"); RES.mkdir(parents=True, exist_ok=True)
    for run in a.runs:
        o = gapood(run, a.seeds, dev); C.dump(RES / f"gapood_{run}.json", o)
        for sp in ("L", "S"):
            for key in ("k1", "k3"):
                d = o[sp][key]  # (per-clip ensemble predictions saved under o["_preds"]); sd = [d[f"s{s}"]["entry_acc"] for s in a.seeds if f"s{s}" in d]
                print(f"{run:24s} gap-OOD {sp} {key}: ENTRY ens {d['ens']['entry_acc']:.3f} (seeds {sd}) bias {d['ens']['bias_s']:+.2f}s score {d['ens']['score']:.3f} "
                      f"| prior {d['prior']['entry_acc']:.3f} bias {d['prior']['bias_s']:+.2f}s", flush=True)
