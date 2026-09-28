"""Stage2_experiments qualitative examples + 'pulled toward COLLISION' rates (native condition, seed-0 arms).
  rescue:  gap >= 1.5 s, ARM hits, B0 and B1 miss
  actor failure: the predicted actor's chain lacks the Exp A oracle vehicle and ARM misses (actor arm)
  pulled: ENTRY miss with the prediction in the later half of [GT ENTRY, GT COLLISION] or later (rate over long-gap clips per arm)
  dynamic labels: D2 last-epoch target shifts (native views)
Usage: python -m stage2.actor.qual BEST_ARM ACTOR_ARM ARM [ARM ...]
"""
from __future__ import annotations

import json, sys

import numpy as np

from stage2.long_context_v2_experiments import common as C
from .report import load, B0, B1, RES


def main():
    best, actor, *others = sys.argv[1:]; fps = C.fps_table()
    P = {a: {q["sample_id"]: q for q in load(a)["preds"]["k1"]} for a in {best, actor, B0, B1, *others}}
    err = lambda q: (q["entry_frame"] - q["entry_gt"]) / fps[q["sample_id"]]; hit = lambda q: abs(err(q)) <= .300001
    gap = lambda q: (q["collision_gt"] - q["entry_gt"]) / fps[q["sample_id"]]
    diag = {x["sid"]: x for x in json.loads((RES / "diag_actor.json").read_text())}
    print(f"## long-gap rescues by {best} (B0, B1 miss)")
    for sid, q in P[best].items():
        if gap(q) >= 1.5 and hit(q) and not hit(P[B0][sid]) and not hit(P[B1][sid]):
            print(f"  {sid} gap {gap(q):.2f}s | {best} err {err(q):+.2f}s | B0 {err(P[B0][sid]):+.2f}s | B1 {err(P[B1][sid]):+.2f}s")
    print(f"## actor-selection failures that {actor} misses")
    xs = [(sid, q) for sid, q in P[actor].items() if diag[sid]["pred"]["oracle_in_chain"] is False and not hit(q)]
    for sid, q in xs[:8]:
        d = diag[sid]["pred"]; print(f"  {sid} gap {gap(q):.2f}s err {err(q):+.2f}s | margin {d['margin']:.2f} | actor at ENTRY {d['at_entry']} | pred==GT-collision actor {diag[sid]['pred_eq_gt']}")
    print(f"  ({len(xs)} such clips)")
    print("## ENTRY pulled toward COLLISION (long-gap clips >= 1.5 s: misses predicted in the later half of the ENTRY..COLLISION span or later)")
    for a in [B0, B1, best, actor, *others]:
        lg = [q for q in P[a].values() if gap(q) >= 1.5]
        pulled = [q for q in lg if not hit(q) and q["entry_frame"] >= q["entry_gt"] + 0.5 * (q["collision_gt"] - q["entry_gt"])]
        print(f"  {a:26s} {len(pulled)}/{len(lg)} = {len(pulled) / len(lg):.2f}" + (f"  e.g. {pulled[0]['sample_id']} err {err(pulled[0]):+.2f}s gap {gap(pulled[0]):.2f}s" if pulled else ""))
    print("## dynamic-label shifts (D2, last epoch, native view)")
    for f in range(5):
        d = json.loads((C.REPO / f"stage2/generalization/results/D2o/cv/fold{f}_seed0/dyn_label_last_epoch.json").read_text())
        big = [(k.split("|")[0], v) for k, v in d.items() if v != 0][:3]
        print(f"  fold {f}: " + ", ".join(f"{s} {v:+d}" for s, v in big))


if __name__ == "__main__": main()
