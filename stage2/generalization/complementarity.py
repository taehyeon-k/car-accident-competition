"""Narrowing campaign, Experiment E — complementarity of an ENTRY candidate w.r.t. the v8 ensemble and its families.

Uses the per-clip ensemble predictions saved by entry_suite (results/entry_suite/<arm>.json; conditions k1/k2/k3/crop25) and, when
available, gap-OOD-L predictions (results/entry_suite/gapood_<run>.json, key _preds). ENTRY hit = official 0.3 s rule.
  rescue = candidate hit AND v8 ensemble miss;  harm = candidate miss AND v8 ensemble hit;  net = rescue - harm
Slices: long-gap (>= 1.5 s; native + 1/2 + 1/3 pooled), native all, crop25, gap-OOD-L (native, 1/3), per source (native).
Error relation: phi correlation of ENTRY-miss indicators and disagreement rate vs E4_sa, E2_sa, XN4_sa (native + 1/3 pooled).
Usage: python -m stage2.generalization.complementarity CAND [CAND ...] [--v8 E4_sa+E2_sa+XN4_sa] [--gapood GOOD_x ...]
"""
from __future__ import annotations

import argparse, json

import numpy as np

from stage2.long_context_v2_experiments import common as C

S = C.REPO / "stage2/generalization/results/entry_suite"


def load(arm):
    return json.loads((S / f"{arm}.json").read_text())["preds"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cands", nargs="+"); ap.add_argument("--v8", default="E4_sa+E2_sa+XN4_sa")
    ap.add_argument("--gapood", nargs="*", default=[], help="gap-OOD run names paired with cands (same order); v8 gap-OOD families are combined")
    a = ap.parse_args(); fps = C.fps_table()
    hit = lambda q: abs(q["entry_frame"] - q["entry_gt"]) / fps[q["sample_id"]] <= .300001
    gap = lambda q: (q["collision_gt"] - q["entry_gt"]) / fps[q["sample_id"]]
    v8 = load(a.v8); fam = {f: load(f) for f in ("E4_sa", "E2_sa", "XN4_sa") if (S / f"{f}.json").exists()}
    idx = lambda P, c: {q["sample_id"]: q for q in P[c]}
    res = {}
    for ci, cand in enumerate(a.cands):
        P = load(cand); r = {}
        def rh(conds, filt=lambda q: True):
            re = ha = n = 0
            for c in conds:
                A, B = idx(P, c), idx(v8, c)
                for sid, q in A.items():
                    if sid not in B or not filt(q): continue
                    x, y = hit(q), hit(B[sid]); n += 1; re += x and not y; ha += (not x) and y
            return {"n": n, "rescue": re, "harm": ha, "net": re - ha}
        r["long_gap"] = rh(("k1", "k2", "k3"), lambda q: gap(q) >= 1.5)
        r["gap>2.5"] = rh(("k1", "k2", "k3"), lambda q: gap(q) >= 2.5)
        r["native"] = rh(("k1",)); r["third"] = rh(("k3",)); r["crop25"] = rh(("crop25",))
        for s_ in ("AIHUB", "CCD", "MMAU", "NEXAR"): r[s_] = rh(("k1",), lambda q, s_=s_: C.source(q) == s_)
        if ci < len(a.gapood):
            G = json.loads((S / f"gapood_{a.gapood[ci]}.json").read_text())["_preds"]["L"]
            Gv = {}
            for f_ in ("GOOD_E4_sa", "GOOD_E2_sa", "GOOD_XN4_sa"):
                pth = S / f"gapood_{f_}.json"
                if pth.exists(): Gv[f_] = json.loads(pth.read_text())["_preds"]["L"]
            if "GOOD_v8" in [p.stem.replace("gapood_", "") for p in S.glob("gapood_GOOD_v8*.json")]: pass
            ref = json.loads((S / "gapood_GOOD_v8.json").read_text())["_preds"]["L"] if (S / "gapood_GOOD_v8.json").exists() else Gv.get("GOOD_E4_sa")
            for key in ("k1", "k3"):
                A = {q["sample_id"]: q for q in G[key]}; B = {q["sample_id"]: q for q in ref[key]}
                re = sum(hit(A[s]) and not hit(B[s]) for s in A if s in B); ha = sum((not hit(A[s])) and hit(B[s]) for s in A if s in B)
                r[f"gapoodL_{key}"] = {"n": len(A), "rescue": re, "harm": ha, "net": re - ha}
        rel = {}
        for f_, Pf in fam.items():
            m1, m2 = [], []
            for c in ("k1", "k3"):
                A, B = idx(P, c), idx(Pf, c)
                for sid, q in A.items():
                    if sid in B: m1.append(not hit(q)); m2.append(not hit(B[sid]))
            m1, m2 = np.array(m1, float), np.array(m2, float)
            rel[f_] = {"phi_miss": round(float(np.corrcoef(m1, m2)[0, 1]), 3), "disagree": round(float(np.mean(m1 != m2)), 3)}
        r["error_relation"] = rel; res[cand] = r
        fmt = lambda d: f"+{d['rescue']}/-{d['harm']} (net {d['net']:+d})"
        print(f"{cand:30s} long-gap {fmt(r['long_gap'])} | >2.5s {fmt(r['gap>2.5'])} | native {fmt(r['native'])} | 1/3 {fmt(r['third'])} | crop25 {fmt(r['crop25'])}")
        print(" " * 31 + " | ".join(f"{s_} {fmt(r[s_])}" for s_ in ("AIHUB", "CCD", "MMAU", "NEXAR"))
              + ("".join(f" | gapOOD-L {k} {fmt(r[f'gapoodL_{k}'])}" for k in ("k1", "k3") if f"gapoodL_{k}" in r)))
        print(" " * 31 + "error relation (phi of misses / disagreement): " + " ".join(f"{f_} {v['phi_miss']}/{v['disagree']}" for f_, v in rel.items()))
    C.dump(C.REPO / "stage2/generalization/results/complementarity.json", res)


if __name__ == "__main__": main()
