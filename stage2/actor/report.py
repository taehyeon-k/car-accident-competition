"""Stage2_experiments screening summary (single seed, 5-fold duplicate-clean CV, 284 clips) from results/entry_suite/<arm>.json.

Baselines are seed-0 members of existing runs (B0 = v8 E4_sa+E2_sa+XN4_sa seed 0, B1 = ODS seed 0; re-evaluated under the tag '@s0').
Per arm: score native / 1/2 / 1/3, ENTRY native / 1/3, COLLISION native, ENTRY by gap (pooled rates), long-gap median bias, crop25 /
crop50 ENTRY and position slope, mean ENTRY peaks (native), ENTRY rescue / harm vs B0 and vs B1 (native + 1/3), and for actor arms
native ENTRY conditional on actor success (predicted actor present at GT ENTRY and the Exp A oracle vehicle in its chain).
Usage: python -m stage2.actor.report ARM [ARM ...]  (prints markdown; writes results/report.json)
"""
from __future__ import annotations

import json, sys

import numpy as np

from stage2.long_context_v2_experiments import common as C

S = C.REPO / "stage2/generalization/results/entry_suite"; RES = C.REPO / "stage2/actor/results"
B0, B1 = "E4_sa+E2_sa+XN4_sa@s0", "ODS@s0"


def load(a): return json.loads((S / f"{a}.json").read_text())


def main():
    arms = [B0, B1] + sys.argv[1:]; fps = C.fps_table()
    hit = lambda q: abs(q["entry_frame"] - q["entry_gt"]) / fps[q["sample_id"]] <= .300001
    diag = {x["sid"]: x for x in json.loads((RES / "diag_actor.json").read_text())}
    ok = lambda sid: diag[sid]["pred"]["at_entry"] and bool(diag[sid]["pred"]["oracle_in_chain"])
    D = {a: load(a) for a in arms}; out = {}
    ref = {b: {c: {q["sample_id"]: q for q in D[b]["preds"][c]} for c in ("k1", "k3")} for b in (B0, B1)}
    print("| arm | S nat/½/⅓ | ENTRY nat/⅓ | COLL | ENTRY <.5/.5–1/1–1.5/1.5–2.5/>2.5 s | bias | crop25 E (slope) | crop50 E (slope) | peaks | vs B0 +/− | vs B1 +/− | actor ok/fail |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for a in arms:
        m = D[a]["metrics"]["ensemble"]; P = D[a]["preds"]; g = m["gap"]
        rh = {}
        for b in (B0, B1):
            re = ha = 0
            for c in ("k1", "k3"):
                for q in P[c]:
                    r_ = ref[b][c].get(q["sample_id"])
                    if r_ is None: continue
                    re += hit(q) and not hit(r_); ha += (not hit(q)) and hit(r_)
            rh[b] = (re, ha)
        pk = [q["_npk"] for q in P["k1"] if "_npk" in q]
        okh = [hit(q) for q in P["k1"] if ok(q["sample_id"])]; fh = [hit(q) for q in P["k1"] if not ok(q["sample_id"])]
        row = {"S": [m[c]["score"] for c in ("k1", "k2", "k3")], "E": [m["k1"]["entry_acc"], m["k3"]["entry_acc"]], "C": m["k1"]["collision_acc"],
               "gap": [g[b]["entry_acc"] for b in g], "bias": m["long_gap_bias_median_s"], "crop25": [m["crop25"]["entry_acc"], m["crop25"]["slope"]],
               "crop50": [m["crop50"]["entry_acc"], m["crop50"]["slope"]], "peaks": float(np.mean(pk)) if pk else None, "rh_B0": rh[B0], "rh_B1": rh[B1],
               "actor_ok": float(np.mean(okh)), "actor_fail": float(np.mean(fh)), "n_ok": len(okh)}
        out[a] = row
        print(f"| {a} | {row['S'][0]:.3f}/{row['S'][1]:.3f}/{row['S'][2]:.3f} | {row['E'][0]:.3f}/{row['E'][1]:.3f} | {row['C']:.3f} | "
              + "/".join(f"{x:.2f}" for x in row["gap"]) + f" | {row['bias']:+.2f} s | {row['crop25'][0]:.3f} ({row['crop25'][1]:.2f}) | "
              f"{row['crop50'][0]:.3f} ({row['crop50'][1]:.2f}) | {row['peaks'] if row['peaks'] is None else round(row['peaks'], 2)} | "
              f"+{rh[B0][0]}/−{rh[B0][1]} | +{rh[B1][0]}/−{rh[B1][1]} | {row['actor_ok']:.3f}/{row['actor_fail']:.3f} |")
    C.dump(RES / "report.json", out)


if __name__ == "__main__": main()
