"""Stage2_experiments D1 diagnostics (no training). Per duplicate-clean clip:
  * actor availability at GT ENTRY (present, detected or interpolated, within +-2 native frames), continuity ENTRY->COLLISION (fraction
    of frames with the actor present), selection margin, agreement with the Exp A oracle ENTRY vehicle (its track id in the chain) and
    agreement between the predicted-COLLISION actor (v8_k1) and the GT-COLLISION actor.
  * ORACLE change-point probe (DIAGNOSTIC ONLY, uses GT COLLISION, never mixed with deployable results): GT-COLLISION actor ->
    backward lane-intrusion depth trajectory (5-frame median) -> ENTRY = last crossing of depth 0 before COLLISION (outside -> inside
    the ego lane); if there is none, the frame of the largest 3-frame depth rise before COLLISION; if the lane is never defined, the
    frame of the largest 3-frame log-area rise.
FPS only for the 0.3 s metric and gap bins. Output stage2/actor/results/diag_actor.json (+ printed summary).
"""
from __future__ import annotations

import json

import numpy as np
from scipy.ndimage import median_filter

from stage2.long_context_v2_experiments import common as C
from stage2.generalization.clean_eval import EXCL
from stage2.objtrack.objlane_features import lane_at

import os
A = C.REPO / os.environ.get("ACTOR_DIR", "stage2/actor/cache_actor"); CORR = C.REPO / "stage2/objtrack/cache_corridor"
RES = C.REPO / "stage2/actor/results"; SUITE = C.REPO / "stage2/generalization/results/entry_suite"
GAPS = [(0, .5), (.5, 1), (1, 1.5), (1.5, 2.5), (2.5, 99)]


def depth_series(a, corr):
    n = len(a["present"]); d = np.full(n, np.nan); la = np.full(n, np.nan)
    for f in range(n):
        if not a["present"][f]: continue
        x1, y1, x2, y2 = [float(v) for v in a["box"][f]]; cx = (x1 + x2) / 2; la[f] = np.log(max((x2 - x1) * (y2 - y1), 1e-5))
        lane = lane_at(corr["left"][f].astype(np.float32), corr["right"][f].astype(np.float32), corr["row_y"], y2)
        if lane:
            Lb, Rb = lane; cc, hw = (Lb + Rb) / 2, (Rb - Lb) / 2; d[f] = np.clip(((x2 - Lb) if cx < cc else (Rb - x1)) / hw, -3, 3)
    return d, la


def fill_med(x, k=5):
    ok = np.isfinite(x)
    if ok.sum() < 3: return x
    y = np.interp(np.arange(len(x)), np.flatnonzero(ok), x[ok]); y = median_filter(y, size=k, mode="nearest"); y[~ok] = np.nan
    return y


def changepoint(a, corr, c):
    d, la = depth_series(a, corr); d, la = fill_med(d), fill_med(la); lo = 1
    ok = [f for f in range(lo, c) if np.isfinite(d[f]) and np.isfinite(d[f - 1])]
    cross = [f for f in ok if d[f - 1] < 0 <= d[f]]
    if cross: return cross[-1], "cross"
    rise = [(d[f] - d[f - 3], f) for f in range(3, c) if np.isfinite(d[f]) and np.isfinite(d[f - 3])]
    if rise: return max(rise)[1], "depth_rise"
    rise = [(la[f] - la[f - 3], f) for f in range(3, c) if np.isfinite(la[f]) and np.isfinite(la[f - 3])]
    if rise: return max(rise)[1], "area_rise"
    return None, "none"


def main():
    fps = C.fps_table(); RES.mkdir(parents=True, exist_ok=True)
    orc = {x["sid"]: x["oracle"] for x in json.loads((C.REPO / "stage2/generalization/results/diag_selection.json").read_text())}
    suite = {k: {p["sample_id"]: p for p in json.loads((SUITE / f"{k}.json").read_text())["preds"]["k1"]} for k in ("E4_sa+E2_sa+XN4_sa", "ODS")}
    recs = []
    for r in C.rows("all"):
        sid = r["sample_id"]
        if sid in EXCL: continue
        e, c, f_ = int(r["entry_frame"]), int(r["collision_frame"]), fps[sid]; corr = np.load(CORR / f"{sid}.npz")
        g, p = np.load(A / "gt" / f"{sid}.npz"), np.load(A / "v8_k1" / f"{sid}.npz"); n = len(g["present"])
        rec = {"sid": sid, "source": C.source(r), "gap_s": (c - e) / f_}
        for name, a in (("gt", g), ("pred", p)):
            chain = a["chain"].tolist()
            rec[name] = {"found": bool(chain), "at_entry": bool(a["present"][max(0, e - 2):e + 3].any()),
                         "det_at_entry": bool((a["present"][max(0, e - 2):e + 3] == 1).any()),
                         "continuity": float((a["present"][e:c + 1] > 0).mean()) if c >= e else 0.0,
                         "margin": float(a["margin"]), "oracle_in_chain": (orc.get(sid) in chain) if orc.get(sid) is not None else None,
                         "chain_len": len(chain)}
        rec["pred_eq_gt"] = bool(g["chain"].size and p["chain"].size and int(g["chain"][0]) == int(p["chain"][0]))
        est, how = changepoint(g, corr, min(c, n - 1))
        rec["cp"] = {"est": est, "how": how, "hit": bool(est is not None and abs(est - e) / f_ <= .300001), "err_s": None if est is None else (est - e) / f_}
        for k, P in suite.items():
            q = P.get(sid); rec[k + "_hit"] = bool(q and abs(q["entry_frame"] - q["entry_gt"]) / f_ <= .300001)
        recs.append(rec)
    C.dump(RES / "diag_actor.json", recs)
    m = lambda xs: round(float(np.mean(xs)), 3) if len(xs) else float("nan")
    print(f"clips {len(recs)}")
    for name in ("gt", "pred"):
        xs = [x[name] for x in recs]; oc = [x["oracle_in_chain"] for x in xs if x["oracle_in_chain"] is not None]
        print(f"  actor[{name}] found {m([x['found'] for x in xs])} | at GT ENTRY {m([x['at_entry'] for x in xs])} (detected {m([x['det_at_entry'] for x in xs])})"
              f" | continuity E->C {m([x['continuity'] for x in xs])} | oracle vehicle in chain {m(oc)} (n {len(oc)}) | margin median {np.median([x['margin'] for x in xs]):.2f}")
    print(f"  predicted-COLLISION actor == GT-COLLISION actor: {m([x['pred_eq_gt'] for x in recs])}")
    print("  ORACLE change-point ENTRY (diagnostic, GT COLLISION):", f"hit {m([x['cp']['hit'] for x in recs])}",
          {h: sum(x['cp']['how'] == h for x in recs) for h in ("cross", "depth_rise", "area_rise", "none")})
    for lo, hi in GAPS:
        xs = [x for x in recs if lo <= x["gap_s"] < hi]
        errs = [x["cp"]["err_s"] for x in xs if x["cp"]["err_s"] is not None]
        print(f"    gap {lo}-{hi}s n {len(xs):3d}: CP {m([x['cp']['hit'] for x in xs])} (median err {np.median(errs):+.2f}s) | v8 {m([x['E4_sa+E2_sa+XN4_sa_hit'] for x in xs])} | ODS {m([x['ODS_hit'] for x in xs])}")
    for lab, f in (("actor ok (pred actor at ENTRY & oracle in chain)", lambda x: x["pred"]["at_entry"] and x["pred"]["oracle_in_chain"]),
                   ("actor failed", lambda x: not (x["pred"]["at_entry"] and x["pred"]["oracle_in_chain"]))):
        xs = [x for x in recs if f(x)]
        print(f"  {lab}: n {len(xs)} | ODS ENTRY {m([x['ODS_hit'] for x in xs])} | v8 {m([x['E4_sa+E2_sa+XN4_sa_hit'] for x in xs])} | CP {m([x['cp']['hit'] for x in xs])}")
    both = [x for x in recs if x["cp"]["hit"] and not x["E4_sa+E2_sa+XN4_sa_hit"]]; rev = [x for x in recs if not x["cp"]["hit"] and x["E4_sa+E2_sa+XN4_sa_hit"]]
    print(f"  CP rescue vs v8: +{len(both)} / -{len(rev)}")


if __name__ == "__main__": main()
