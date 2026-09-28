"""Weakness analysis of v8 + D5H (3-seed out-of-fold predictions, duplicate-clean 284 clips, all suite conditions incl. the DACON-like
view). Compares with v8 (same seeds, same conditions). FPS only for the 0.3 s metric and second-based bins.
Usage: python -m stage2.actor.weakness  -> prints markdown, writes stage2/actor/results/weakness.json
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np

from stage2.long_context_v2_experiments import common as C

S = C.REPO / "stage2/generalization/results/entry_suite"; RES = C.REPO / "stage2/actor/results"
ARMS = {"v8": "E4_sa+E2_sa+XN4_sa@s012d", "v8+D5H": "E4_sa+E2_sa+XN4_sa+D5H@s012d"}
CONDS = ["k1", "k2", "k3", "crop50", "crop25", "dacon"]
SRC = ("AIHUB", "CCD", "MMAU", "NEXAR")


def f1(y, p):
    return C.macro_f1(y, p)


def main():
    fps = C.fps_table(); rows = {r["sample_id"]: r for r in C.rows("all")}
    D = {a: json.loads((S / f"{k}.json").read_text()) for a, k in ARMS.items()}
    diag = {x["sid"]: x for x in json.loads((RES / "diag_actor.json").read_text())}
    sec = lambda q, e: (q[f"{e}_frame"] - q[f"{e}_gt"]) / fps[q["sample_id"]]
    hit = lambda q, e: abs(sec(q, e)) <= .300001
    gap = lambda q: (q["collision_gt"] - q["entry_gt"]) / fps[q["sample_id"]]
    out = {}; pr = print
    m = lambda xs: float(np.mean(xs)) if len(xs) else float("nan")

    pr("## 1. Points lost per component (score = .35 ENTRY + .35 COLL + .15 side F1 + .15 evasion F1)\n")
    pr("| arm | cond | score | lost: ENTRY | COLL | side | evasion |\n|---|---|---|---|---|---|---|")
    for a in ARMS:
        for c in CONDS:
            mm = C.metrics(D[a]["preds"][c]); lost = [.35 * (1 - mm["entry_acc"]), .35 * (1 - mm["collision_acc"]), .15 * (1 - mm["side_f1"]), .15 * (1 - mm["evasion_f1"])]
            out.setdefault("lost", {}).setdefault(a, {})[c] = lost
            pr(f"| {a} | {c} | {mm['score']:.3f} | {lost[0]:.3f} | {lost[1]:.3f} | {lost[2]:.3f} | {lost[3]:.3f} |")

    P = D["v8+D5H"]["preds"]; V = D["v8"]["preds"]
    pr("\n## 2. By source (v8+D5H; v8 in brackets): ENTRY / COLL / side F1 / evasion F1\n")
    pr("| source | n | cond | ENTRY | COLL | side F1 | evasion F1 |\n|---|---|---|---|---|---|---|")
    for s in SRC:
        for c in ("k1", "k3", "dacon"):
            ps = [q for q in P[c] if C.source(q) == s]; vs = [q for q in V[c] if C.source(q) == s]; a_, b_ = C.metrics(ps), C.metrics(vs)
            pr(f"| {s} | {len(ps)} | {c} | {a_['entry_acc']:.3f} ({b_['entry_acc']:.3f}) | {a_['collision_acc']:.3f} ({b_['collision_acc']:.3f}) | "
               f"{a_['side_f1']:.3f} ({b_['side_f1']:.3f}) | {a_['evasion_f1']:.3f} ({b_['evasion_f1']:.3f}) |")

    pr("\n## 3. ENTRY error structure (v8+D5H)\n")
    for c in ("k1", "k3", "dacon"):
        miss = [q for q in P[c] if not hit(q, "entry")]; e = np.array([sec(q, "entry") for q in miss])
        pulled = [q for q in miss if q["entry_frame"] >= q["entry_gt"] + .5 * (q["collision_gt"] - q["entry_gt"]) and gap(q) > .3]
        pr(f"* {c}: misses {len(miss)}/{len(P[c])} | late {np.mean(e > 0):.2f} / early {np.mean(e < 0):.2f} | median |err| {np.median(np.abs(e)):.2f}s | "
           f"near misses 0.3-0.6s {np.mean(np.abs(e) <= .6):.2f} | >1s {np.mean(np.abs(e) > 1):.2f} | >2s {np.mean(np.abs(e) > 2):.2f} | "
           f"pulled toward COLLISION {len(pulled)} | ENTRY predicted at/after COLLISION {sum(q['entry_frame'] >= q['collision_gt'] for q in miss)}")
    pr("\n| gap bin | n | ENTRY k1 | ENTRY k3 | ENTRY dacon | median signed err (misses, k1) | v8 ENTRY k1 / k3 / dacon |\n|---|---|---|---|---|---|---|")
    for lo, hi in ((0, .5), (.5, 1), (1, 1.5), (1.5, 2.5), (2.5, 99)):
        sel = lambda X: [q for q in X if lo <= gap(q) < hi]
        mk = [sec(q, "entry") for q in sel(P["k1"]) if not hit(q, "entry")]
        pr(f"| {lo}-{hi} s | {len(sel(P['k1']))} | " + " | ".join(f"{m([hit(q, 'entry') for q in sel(P[c])]):.3f}" for c in ("k1", "k3", "dacon"))
           + f" | {np.median(mk) if mk else float('nan'):+.2f}s | " + "/".join(f"{m([hit(q, 'entry') for q in sel(V[c])]):.3f}" for c in ("k1", "k3", "dacon")) + " |")

    pr("\n## 4. ENTRY vs position in the clip and clip length (native, v8+D5H)\n")
    rel = lambda q: (q["entry_gt"] - q["_lo"]) / max(q["_hi"] - q["_lo"], 1)
    for lo, hi in ((0, .3), (.3, .5), (.5, .7), (.7, .85), (.85, 1.01)):
        xs = [q for q in P["k1"] if lo <= rel(q) < hi]; ys = [q for q in P["crop25"] if lo <= rel(q) < hi]
        pr(f"* GT ENTRY at {lo:.2f}-{hi:.2f} of the clip: native n {len(xs)} ENTRY {m([hit(q, 'entry') for q in xs]):.3f} | crop25 n {len(ys)} ENTRY {m([hit(q, 'entry') for q in ys]):.3f}")
    L = lambda q: int(rows[q["sample_id"]]["num_frames"])
    for lo, hi in ((0, 150), (150, 300), (300, 750), (750, 10 ** 6)):
        xs = [q for q in P["k1"] if lo <= L(q) < hi]
        pr(f"* clip length {lo}-{hi} frames: n {len(xs)} ENTRY {m([hit(q, 'entry') for q in xs]):.3f} COLL {m([hit(q, 'collision') for q in xs]):.3f}")

    pr("\n## 5. COLLISION errors (v8+D5H)\n")
    for c in ("k1", "k3", "dacon"):
        miss = [q for q in P[c] if not hit(q, "collision")]; e = np.array([sec(q, "collision") for q in miss])
        pr(f"* {c}: misses {len(miss)} | late {np.mean(e > 0):.2f} early {np.mean(e < 0):.2f} | median |err| {np.median(np.abs(e)):.2f}s | 0.3-0.6s {np.mean(np.abs(e) <= .6):.2f} | >1s {np.mean(np.abs(e) > 1):.2f} | "
           f"by source " + ", ".join(f"{s} {sum(C.source(q) == s for q in miss)}" for s in SRC))
    both = [q for q in P["k1"] if not hit(q, "collision") and not hit(q, "entry")]
    pr(f"* native: COLLISION misses that also miss ENTRY: {len(both)}/{sum(not hit(q, 'collision') for q in P['k1'])}")

    pr("\n## 6. Attributes (native, v8+D5H)\n")
    for key, gt, pd_ in (("side", "entry_side_gt", "entry_side"), ("evasion", "evasion_gt", "evasion_space")):
        y = [q[gt] for q in P["k1"]]; p = [q[pd_] for q in P["k1"]]
        cm = Counter((a, b) for a, b in zip(y, p))
        wrong = [q for q in P["k1"] if q[gt] != q[pd_]]
        pr(f"* {key}: F1 {f1(y, p):.3f} | confusion (gt,pred) {dict(cm)} | errors with ENTRY miss {m([not hit(q, 'entry') for q in wrong]):.2f} vs overall ENTRY miss {m([not hit(q, 'entry') for q in P['k1']]):.2f}"
               f" | F1 k3 {f1([q[gt] for q in P['k3']], [q[pd_] for q in P['k3']]):.3f} dacon {f1([q[gt] for q in P['dacon']], [q[pd_] for q in P['dacon']]):.3f}")

    pr("\n## 7. Actor dependence (native ENTRY)\n")
    for lab, f in (("actor ok", lambda s: diag[s]["pred"]["at_entry"] and bool(diag[s]["pred"]["oracle_in_chain"])),
                   ("actor wrong / missing", lambda s: not (diag[s]["pred"]["at_entry"] and bool(diag[s]["pred"]["oracle_in_chain"])))):
        xs = [q for q in P["k1"] if f(q["sample_id"])]; vs = [q for q in V["k1"] if f(q["sample_id"])]
        pr(f"* {lab}: n {len(xs)} | v8+D5H {m([hit(q, 'entry') for q in xs]):.3f} | v8 {m([hit(q, 'entry') for q in vs]):.3f}")
    mg = np.array([diag[q["sample_id"]]["pred"]["margin"] for q in P["k1"]]); qs = np.quantile(mg, [0, .25, .5, .75, 1])
    for lo, hi in zip(qs[:-1], qs[1:]):
        xs = [q for q in P["k1"] if lo <= diag[q["sample_id"]]["pred"]["margin"] <= hi]; vs = {q["sample_id"] for q in xs}
        pr(f"* ranker margin {lo:.2f}-{hi:.2f}: v8+D5H {m([hit(q, 'entry') for q in xs]):.3f} | v8 {m([hit(q, 'entry') for q in V['k1'] if q['sample_id'] in vs]):.3f}")

    pr("\n## 8. Seeds (per-seed ENTRY native / 1/3 / dacon, score native)\n")
    for a in ARMS:
        ps = D[a]["metrics"]["per_seed"]
        pr(f"* {a}: " + " | ".join(f"{s}: E {v['k1']['entry_acc']:.3f}/{v['k3']['entry_acc']:.3f}/{v['dacon']['entry_acc']:.3f} S {v['k1']['score']:.3f}" for s, v in ps.items()))

    pr("\n## 9. Hard core: ENTRY missed under every condition (v8+D5H)\n")
    idx = {c: {q["sample_id"]: q for q in P[c]} for c in CONDS}
    core = [s for s in idx["k1"] if all(not hit(idx[c][s], "entry") for c in CONDS if s in idx[c])]
    easy = [s for s in idx["k1"] if all(hit(idx[c][s], "entry") for c in CONDS if s in idx[c])]
    pr(f"* {len(core)} clips ({len(core) / len(idx['k1']):.2f}) missed everywhere; {len(easy)} always hit")
    pr(f"* hard core by source {dict(Counter(C.source(idx['k1'][s]) for s in core))} (all: {dict(Counter(C.source(q) for q in P['k1']))})")
    pr(f"* hard core median gap {np.median([gap(idx['k1'][s]) for s in core]):.2f}s vs all {np.median([gap(q) for q in P['k1']]):.2f}s | "
       f"median signed err native {np.median([sec(idx['k1'][s], 'entry') for s in core]):+.2f}s | late share {np.mean([sec(idx['k1'][s], 'entry') > 0 for s in core]):.2f} | "
       f"actor ok {np.mean([diag[s]['pred']['at_entry'] and bool(diag[s]['pred']['oracle_in_chain']) for s in core]):.2f} vs all {np.mean([diag[s]['pred']['at_entry'] and bool(diag[s]['pred']['oracle_in_chain']) for s in idx['k1']]):.2f} | "
       f"COLL also missed (native) {np.mean([not hit(idx['k1'][s], 'collision') for s in core]):.2f}")
    pr("* examples: " + ", ".join(f"{s} (gap {gap(idx['k1'][s]):.1f}s, err {sec(idx['k1'][s], 'entry'):+.1f}s)" for s in core[:8]))
    out["core"] = core
    C.dump(RES / "weakness.json", out)


if __name__ == "__main__": main()
