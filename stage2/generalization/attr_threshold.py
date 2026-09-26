"""Label-free test-time threshold adaptation for the binary attributes (side, evasion).
Held-out sources shift the *level* of the attribute probabilities (evasion AUC stays 0.6-0.8 but mean p moves by up to +-0.2),
so a fixed 0.5 threshold loses macro-F1. Rules, all computed on the unlabelled test set only (no GT, no FPS):
  fixed   : p >= 0.5
  median  : p >= median(p over the test set)                      (assumes ~balanced classes)
  trainq  : p >= quantile(p_test, 1 - pi_train)                   (pi_train = positive rate in the training sources)
  logitc  : centre the test logits (subtract mean logit), threshold 0  (= median-like but mean based)
Evaluated per held-out source on LOSO predictions (seed-ensemble probs) and on CV folds.
Usage: python -m stage2.generalization.attr_threshold RUN [RUN ...]
"""
from __future__ import annotations

import argparse, collections, glob, json

import numpy as np

from stage2.long_context_v2_experiments import common as C

R = C.REPO / "stage2/generalization/results"


def macro_f1(y, p): return C.macro_f1(list(map(int, y)), list(map(int, p)))


def rules(p, pi):
    lg = np.log(p + 1e-6) - np.log(1 - p + 1e-6)
    return {"fixed": p >= .5, "median": p >= np.median(p), "trainq": p >= np.quantile(p, 1 - pi), "logitc": lg - lg.mean() >= 0}


def collect(files):
    per = collections.defaultdict(list)
    for f in files:
        for p in json.load(open(f)): per[p["sample_id"]].append(p)
    out = []
    for sid, ps in per.items():
        out.append({"src": ps[0]["source_id"].split(":")[0], "eva": np.mean([x["evasion_prob"] for x in ps]), "side": np.mean([x["side_prob_right"] for x in ps]),
                    "eva_gt": ps[0]["evasion_gt"], "side_gt": ps[0]["entry_side_gt"]})
    return out


def prior(src_excl, key):
    rows = [r for r in C.rows("all") if r["source_id"].split(":")[0] not in src_excl]
    return np.mean([int(r["evasion_space"]) if key == "eva" else int(r["entry_side"] == "RIGHT") for r in rows])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); a = ap.parse_args()
    for run in a.runs:
        loso = not (R / run / "cv").exists()
        if loso:  # held-out source = the test set
            for key in ("eva", "side"):
                res = collections.defaultdict(list)
                for src in ("AIHUB", "CCD", "MMAU", "NEXAR"):
                    d = collect(glob.glob(str(R / run / f"{src}_seed*/predictions.json")))
                    d = [x for x in d if x["src"] == src]; p = np.array([x[key] for x in d]); y = np.array([x[f"{key}_gt"] for x in d])
                    for k, v in rules(p, prior({src}, key)).items(): res[k].append(macro_f1(y, v))
                print(f"{run} LOSO {key}: " + " | ".join(f"{k} " + "/".join(f"{x:.3f}" for x in v) + f" mean {np.mean(v):.3f}" for k, v in res.items()))
        else:  # CV: each val fold is a test set (mixed sources)
            for key in ("eva", "side"):
                res = collections.defaultdict(list)
                for fold in range(5):
                    d = collect(glob.glob(str(R / run / f"cv/fold{fold}_seed*/predictions.json")))
                    p = np.array([x[key] for x in d]); y = np.array([x[f"{key}_gt"] for x in d])
                    for k, v in rules(p, prior(set(), key)).items(): res[k].append(macro_f1(y, v))
                print(f"{run} CV {key}: " + " | ".join(f"{k} {np.mean(v):.3f}" for k, v in res.items()))


if __name__ == "__main__": main()
