"""v5 ensemble test: C0_avg + X_ema + M_motion (unchanged) + one 4th family, same ensemble size.

CV: pooled OOF, probability-averaged over all members (every run x CV seed) per fold, then the unchanged v5 decoding
(decode_motion.decode: camera-shift fusion on COLLISION with beta chosen per fold on the other folds, native snap, no prior).
Fixed split: members' fixed-split checkpoints, beta frozen at 1.0 (as fixed_eval.py / submission v5).
Old families are read from long_context_v2_experiments/results (predictions.json + oof_full.json), new ones from phase_study.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from stage2.long_context_v2_experiments import common as C
from stage2.long_context_v2_experiments.decode_motion import decode
from stage2.long_context_v2_experiments.ensemble import load as load_old
from stage2.long_context_v2_experiments.fixed_eval import infer
from .model import load as load_new

OLD = C.REPO / "stage2/long_context_v2_experiments/results"
NEW = C.REPO / "stage2/phase_study/results"
BASE = ["C0_avg", "X_ema", "M_motion"]
BETAS = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0]


def root(run): return OLD if (OLD / run).is_dir() else NEW


def members_fold(run, k, seeds=None):
    """[(predictions-with-logits by id, (side_prob, evasion_prob) by id)] for the CV seeds of run on fold k."""
    out = []
    paths = sorted((root(run) / run / "cv").glob(f"fold{k}_seed*/predictions.json"))
    if seeds is not None: paths = [p for p in paths if int(p.parent.name.split("_seed")[1]) in seeds]
    for p in paths:
        preds = {x["sample_id"]: x for x in json.loads(p.read_text())}
        if "side_prob_right" in next(iter(preds.values())):
            probs = {s: (x["side_prob_right"], x["evasion_prob"]) for s, x in preds.items()}
        else:
            full = json.loads((p.parent / "oof_full.json").read_text())
            probs = {x["sample_id"]: (x["side_prob_right"], x["evasion_prob"]) for x in full}
        out.append((preds, probs))
    return out


def lp(x): x = np.asarray(x, float); return x - np.logaddexp.reduce(x)


def line(name, b):
    n = b["source:NEXAR"]; o = b["overall"]
    return (f"{name:34s} all {o['score']:.4f} (E {o['entry_acc']:.3f} C {o['collision_acc']:.3f} side {o['side_f1']:.3f} eva {o['evasion_f1']:.3f}) | "
            f"NEXAR {n['score']:.4f} | >1000 {b['bin:>1000']['score']:.4f} | non-NEXAR {b['source:non-NEXAR']['score']:.4f}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("fourth", nargs="+", help="4th-family run ids (old or phase_study)")
    ap.add_argument("--seed-sets", default="0,1", help="space-separated CV seed draws for the 4th family, e.g. '0,1 2,3'")
    ap.add_argument("--fixed-seeds", default="0,1", help="fixed-split seeds of the 4th family")
    ap.add_argument("--no-fixed", action="store_true"); ap.add_argument("--out", default=None); a = ap.parse_args()
    dev = torch.device("cuda"); res = {}
    draws = [set(int(x) for x in d.split(",")) for d in a.seed_sets.split()]
    for f in a.fourth:
        runs = BASE + [f]; entry = {"draws": []}
        for d in draws:
            if any(not list((root(f) / f / "cv").glob(f"fold0_seed{s_}/predictions.json")) for s_ in d): continue
            b, chosen, sizes, _ = cv_eval_fourth(runs, f, d)
            entry["draws"].append({"seeds": sorted(d), "cv": b, "beta": chosen, "members": sizes})
            print(line(f"CV   +{f} seeds {sorted(d)}", b) + f" | beta {chosen} | members {sizes[0]}", flush=True)
        for key, name in (("overall", "all"), ("source:NEXAR", "NEXAR"), ("bin:>1000", ">1000"), ("source:non-NEXAR", "nonNEX")):
            v = [x["cv"][key]["score"] for x in entry["draws"]]
            entry[f"cv_{name}"] = (float(np.mean(v)), float(np.std(v)), len(v))
        for key in ("entry_acc", "collision_acc", "side_f1", "evasion_f1", "entry_catastrophic", "collision_catastrophic"):
            entry[f"cv_all_{key}"] = float(np.mean([x["cv"]["overall"][key] for x in entry["draws"]]))
        print(f"==> +{f}: CV all {entry['cv_all'][0]:.4f}±{entry['cv_all'][1]:.4f} NEXAR {entry['cv_NEXAR'][0]:.4f}±{entry['cv_NEXAR'][1]:.4f} "
              f">1000 {entry['cv_>1000'][0]:.4f}±{entry['cv_>1000'][1]:.4f} (n draws {entry['cv_all'][2]})", flush=True)
        if not a.no_fixed:
            fb, n = fixed_eval_fourth(runs, f, dev, {int(x) for x in a.fixed_seeds.split(",")}); entry.update(fixed=fb, fixed_members=n)
            print(line(f"FIX  +{f}", fb) + f" | members {n}", flush=True)
        res[f] = entry
    if a.out: C.dump(a.out, res)


def cv_eval_fourth(runs, fourth, seeds):
    folds = {}
    for k in range(5):
        mem = [m for r in runs[:-1] for m in members_fold(r, k)] + members_fold(fourth, k, seeds)
        folds[k] = mem
    pooled = {}
    for k, mem in folds.items():
        merged = []
        for sid in mem[0][0]:
            items = [m[0][sid] for m in mem]
            p = {k_: v for k_, v in items[0].items() if k_ not in ("entry_logits", "collision_logits", "phase_logp")}
            for e in ("entry", "collision"):
                p[f"{e}_logp"] = np.log(np.mean([np.exp(lp(it[f"{e}_logits"])) for it in items], 0) + 1e-12)
            p["entry_side"] = int(np.mean([m[1][sid][0] for m in mem]) >= .5)
            p["evasion_space"] = int(np.mean([m[1][sid][1] for m in mem]) >= .5)
            merged.append(p)
        pooled[k] = merged
    cache = {}
    def preds(k, b):
        if (k, b) not in cache: cache[(k, b)] = [decode(p, None, 0, 0, b, False, True) for p in pooled[k]]
        return cache[(k, b)]
    final, chosen = [], []
    for k in pooled:
        def other(b):
            ps = [q for j in pooled if j != k for q in preds(j, b)]
            return C.metrics([q for q in ps if C.source(q) == "NEXAR"])["collision_acc"] + C.metrics(ps)["score"]
        b = max(BETAS, key=other); chosen.append(b); final += preds(k, b)
    return C.breakdown(final), chosen, [len(folds[k]) for k in folds], final


def fixed_eval_fourth(runs, fourth, dev, seeds):
    models = []
    for r in runs:
        for p in sorted((root(r) / r).glob("seed*/checkpoint.pt")):
            if r == fourth and int(p.parent.name[4:]) not in seeds: continue
            models.append(load_old(p, dev) if root(r) == OLD else load_new(p, dev))
    raw = infer(models, C.rows("val"), dev)
    return C.breakdown([decode(p, None, 0, 0, 1.0, False, True) for p in raw]), len(models)


if __name__ == "__main__": main()
