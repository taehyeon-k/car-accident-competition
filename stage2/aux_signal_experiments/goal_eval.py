"""Evaluate a candidate Stage-2 ensemble against the goal (validation > 0.8 overall and > 0.7 NEXAR).

Both protocols use saved predictions only (no retraining) and the unchanged v5 decoding (camera-shift COLLISION fusion + native snap,
no position prior, direct heads, attributes = mean of member probabilities):
  CV     pooled 5-fold OOF over all 349 clips (80 NEXAR); beta chosen per fold on the OTHER folds (decode_motion criterion)
  fixed  the 279/70 split: every fixed-seed run of each family on the 70 val clips (15 NEXAR); beta frozen at 1.0 (as v5)
Old LC-v2 fixed runs store only argmax attributes; they enter the attribute average as 0/1 votes.
Usage: python -m stage2.aux_signal_experiments.goal_eval FAMILY[:seeds] ...   e.g. M1_both E1_M1_ema:0,1,2
"""
from __future__ import annotations

import argparse, json

import numpy as np

from stage2.long_context_v2_experiments import common as C
from stage2.long_context_v2_experiments.decode_motion import decode
from stage2.phase_study import ensemble_test as PE
from . import ensemble_test as ET  # sets PE.root to search all result roots


def lp(x): x = np.asarray(x, float); return x - np.logaddexp.reduce(x)


def fixed_preds(spec):
    runs = []
    for run, seeds in spec:
        for p in sorted((PE.root(run) / run).glob("seed*/predictions.json")):
            if seeds is not None and int(p.parent.name[4:]) not in seeds: continue
            runs.append({x["sample_id"]: x for x in json.loads(p.read_text())})
    out = []
    for sid in runs[0]:
        items = [r[sid] for r in runs]
        p = {k: v for k, v in items[0].items() if not k.endswith("_logits") and k != "phase_logp"}
        for e in ("entry", "collision"):
            p[f"{e}_logp"] = np.log(np.mean([np.exp(lp(it[f"{e}_logits"])) for it in items], 0) + 1e-12)
        side = [it.get("side_prob_right", float(it["entry_side"])) for it in items]
        eva = [it.get("evasion_prob", float(it["evasion_space"])) for it in items]
        p["entry_side"] = int(np.mean(side) >= .5); p["evasion_space"] = int(np.mean(eva) >= .5)
        out.append(p)
    return [decode(p, None, 0, 0, 1.0, False, True) for p in out], len(runs)


def parse(tokens):
    spec = []
    for t in tokens:
        run, _, seeds = t.partition(":")
        spec.append((run, {int(x) for x in seeds.split(",")} if seeds else None))
    return spec


def line(tag, b, extra=""):
    o, n = b["overall"], b["source:NEXAR"]
    ok = "GOAL MET" if o["score"] > 0.8 and n["score"] > 0.7 else ""
    return (f"{tag:6s} all {o['score']:.4f} (E {o['entry_acc']:.3f} C {o['collision_acc']:.3f} side {o['side_f1']:.3f} eva {o['evasion_f1']:.3f}) "
            f"NEXAR {n['score']:.4f} (E {n['entry_acc']:.3f} C {n['collision_acc']:.3f}) >1000 {b['bin:>1000']['score']:.4f} "
            f"non-NEXAR {b['source:non-NEXAR']['score']:.4f} {extra} {ok}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("families", nargs="+"); ap.add_argument("--out", default=None)
    ap.add_argument("--attr-stack", action="store_true", help="replace side/evasion by the event-window stacker (CV only)"); a = ap.parse_args()
    spec = parse(a.families)
    cv, betas = ET.fused_cv(spec); bcv = C.breakdown(cv)
    if a.attr_stack:
        from .attr_stack import stack
        folds = {}
        for q in cv: folds.setdefault(q["fold"], []).append({**q, "frames": q["frames_kept"]})
        st = stack(folds); bst = C.breakdown(st)
    try:
        fx, n = fixed_preds(spec); bfx = C.breakdown(fx)
    except IndexError:  # some family has no fixed-split runs (CV-only research runs)
        fx, n, bfx = None, 0, None
    members = len(ET.members(spec, 0))
    print(" + ".join(a.families)); print(line("CV", bcv, f"| members/fold {members} beta {betas}"))
    if a.attr_stack: print(line("CV+AS", bst, "| event-window attribute stacker"))
    if bfx is not None: print(line("fixed", bfx, f"| members {n}"))
    if a.out: C.dump(a.out, {"families": a.families, "cv": bcv, "fixed": bfx, "beta": betas})


if __name__ == "__main__": main()
