"""Narrowing campaign, Experiment A — is the ENTRY-relevant vehicle among OB_D's candidates? (diagnostic only; no training)

For every duplicate-clean labelled clip and every tracked vehicle (RF-DETR + ByteTrack, cache_tracks) around GT ENTRY:
  detector confidence, persistence (fraction of frames in ENTRY ± 1 s where the track exists), box area, centrality, growth, ego-lane
  overlap, signed lane-intrusion depth, change of depth, lateral motion toward the lane centre (lane-relative, i.e. camera-compensated)
  and (for the slot-selected tracks) appearance change.
Oracle relevant track (DIAGNOSTIC ONLY — uses GT ENTRY / COLLISION and future frames, never used for inference):
  among tracks present in >= 40 % of frames in [ENTRY - 0.5 s, ENTRY + 0.5 s], maximise
      (depth(E + 0.5 s) - depth(E - 0.5 s)) + 0.5 * max(depth(C), 0) + 0.5 * overlap(C)
  (the vehicle whose ego-lane intrusion rises across ENTRY and that is still intruding at COLLISION).
Current selector = per frame, top-K by area x centrality (obj_crops.per_frame_slots); recall@K at the ENTRY frame (± 2 frames).
Conditional OB_D / v8-E4 ENTRY accuracy (native, 6-seed ensemble predictions from results/entry_suite) by: oracle in top-3 or not,
lane corridor valid at the oracle's contact point, oracle track stable (persistence >= .8) or fragmented.
FPS is used only for the second-based windows and the metric. Output results/diag_selection.json.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from stage2.long_context_v2_experiments import common as C
from stage2.generalization.clean_eval import EXCL
from stage2.objtrack.obj_crops import per_frame_slots, VEH
from stage2.objtrack.objlane_features import lane_at

TR = C.REPO / "stage2/objtrack/cache_tracks"; CORR = C.REPO / "stage2/objtrack/cache_corridor"; CROP = C.REPO / "stage2/objtrack/cache_objcrop"
SUITE = C.REPO / "stage2/generalization/results/entry_suite"


def track_table(d, corr, n):
    """per track: dict frame -> (box, score)."""
    fr, box, sc, cls, tr = d["frame"], d["box"], d["score"], d["cls"], d["track"]
    out = defaultdict(dict)
    for f, b, s, c, t in zip(fr, box, sc, cls, tr):
        if t >= 0 and c in VEH: out[int(t)][int(f)] = (b, float(s))
    return out


def lane_feats(b, corr, f):
    x1, y1, x2, y2 = [float(v) for v in b]; cx = (x1 + x2) / 2
    L, R = corr["left"][f].astype(np.float32), corr["right"][f].astype(np.float32)
    lane = lane_at(L, R, corr["row_y"], y2)
    if not lane: return None
    Lb, Rb = lane; cc, hw = (Lb + Rb) / 2, (Rb - Lb) / 2
    depth = ((x2 - Lb) if cx < cc else (Rb - x1)) / hw
    return {"depth": float(np.clip(depth, -3, 3)), "overlap": max(0.0, min(x2, Rb) - max(x1, Lb)) / max(x2 - x1, 1e-4), "off": float((cx - cc) / hw)}


def nearest(track, f, tol):
    ks = [k for k in track if abs(k - f) <= tol]
    return min(ks, key=lambda k: abs(k - f)) if ks else None


def main():
    fps = C.fps_table(); rows = [r for r in C.rows("all") if r["sample_id"] not in EXCL]
    preds = {}
    for name, key in (("OB_D", "OB_D@s6"), ("E4_sa", "E4_sa@s6"), ("v8", "E4_sa+E2_sa+XN4_sa")):
        p = json.loads((SUITE / f"{key}.json").read_text())["preds"]["k1"]; preds[name] = {q["sample_id"]: q for q in p}
    hit = lambda q: abs(q["entry_frame"] - q["entry_gt"]) / fps[q["sample_id"]] <= .300001
    recs = []
    for r in rows:
        sid = r["sample_id"]; f_ = fps[sid]; d = np.load(TR / f"{sid}.npz"); corr = np.load(CORR / f"{sid}.npz"); n = int(d["nframes"])
        e, c = int(r["entry_frame"]), int(r["collision_frame"]); h = max(1, int(round(0.5 * f_))); w1 = max(1, int(round(1.0 * f_)))
        T = track_table(d, corr, n)
        # oracle
        best, bscore = None, -1e9; cands = []
        big_at_c = set()
        for ff in range(max(0, c - 3), min(n, c + 4)):
            m_ = (d["frame"] == ff) & np.isin(d["cls"], VEH) & (d["track"] >= 0); bx_ = d["box"][m_]; tt_ = d["track"][m_]
            big_at_c |= set(tt_[np.argsort(-(bx_[:, 2] - bx_[:, 0]) * (bx_[:, 3] - bx_[:, 1]))][:2].tolist())
        for t, tr in T.items():
            pres = np.mean([k in tr for k in range(max(0, e - h), min(n, e + h + 1))])
            fe = nearest(tr, e, 2)
            if fe is None: continue
            b, s = tr[fe]; lf = lane_feats(b, corr, fe)
            fa, fb, fc = nearest(tr, e - h, h), nearest(tr, e + h, h), nearest(tr, c, 3)
            la = lane_feats(tr[fa][0], corr, fa) if fa is not None else None; lb = lane_feats(tr[fb][0], corr, fb) if fb is not None else None
            lc = lane_feats(tr[fc][0], corr, fc) if fc is not None else None
            fp = nearest(tr, e - 3, 3)
            x1, y1, x2, y2 = [float(v) for v in b]; area = (x2 - x1) * (y2 - y1); cen = 1 - abs((x1 + x2) / 2 - .5)
            growth = float(np.log(max(area, 1e-5)) - np.log(max((tr[fp][0][2] - tr[fp][0][0]) * (tr[fp][0][3] - tr[fp][0][1]), 1e-5))) if fp is not None and fp != fe else 0.0
            lat = (abs(lane_feats(tr[fp][0], corr, fp)["off"]) - abs(lf["off"])) if (fp is not None and fp != fe and lf and lane_feats(tr[fp][0], corr, fp)) else 0.0
            persist = float(np.mean([k in tr for k in range(max(0, e - w1), min(n, e + w1 + 1))]))
            feat = {"track": t, "conf": s, "persist": persist, "area": area, "centrality": cen, "growth": growth,
                    "lane_valid": bool(lf), "overlap": lf["overlap"] if lf else None, "depth": lf["depth"] if lf else None,
                    "d_depth": (lb["depth"] - la["depth"]) if (la and lb) else None, "lateral": lat}
            cands.append(feat)
            feat["partner_like"] = t in big_at_c
            # oracle v2: the track must be the collision partner-like vehicle: present within 3 frames of COLLISION and among the two
            # largest vehicle boxes there (v1 picked distant vehicles straight ahead whose far-field corridor "intrusion" is noise)
            if fc is None or t not in big_at_c: continue
            if pres >= 0.4 and la and lb:
                sc = (lb["depth"] - la["depth"]) + 0.5 * max(lc["depth"], 0) + 0.5 * lc["overlap"] if lc else (lb["depth"] - la["depth"])
                if sc > bscore: best, bscore = t, sc
        # current selector ranks at ENTRY (per-frame area x centrality)
        pres, boxes, tids = per_frame_slots(d, n)
        ranks = []
        for ff in range(max(0, e - 2), min(n, e + 3)):
            fr_m = (d["frame"] == ff) & np.isin(d["cls"], VEH); bx = d["box"][fr_m]; tt = d["track"][fr_m]
            sc_ = (bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1]) * (1 - np.abs((bx[:, 0] + bx[:, 2]) / 2 - .5))
            order = tt[np.argsort(-sc_)]
            ranks.append(int(np.nonzero(order == best)[0][0]) + 1 if best is not None and best in order else 99)
        rank = min(ranks) if ranks else 99
        orc = next((x for x in cands if x["track"] == best), None)
        recs.append({"sid": sid, "source": C.source(r), "gap_s": (c - e) / f_, "n_tracks_at_E": len(cands), "oracle": best, "oracle_score": bscore,
                     "oracle_rank": rank, "oracle_feat": orc, "OB_D_hit": hit(preds["OB_D"][sid]), "E4_hit": hit(preds["E4_sa"][sid]),
                     "v8_hit": hit(preds["v8"][sid]), "candidates": cands})
    C.dump(C.REPO / "stage2/generalization/results/diag_selection.json", recs)
    has = [x for x in recs if x["oracle"] is not None]
    print(f"clips {len(recs)}, oracle found {len(has)} ({len(has)/len(recs):.2f}); mean tracks at ENTRY {np.mean([x['n_tracks_at_E'] for x in recs]):.1f}")
    rec = lambda xs, K: float(np.mean([x["oracle_rank"] <= K for x in xs])) if xs else float("nan")
    print("selector recall@1/3/5 (all):", [round(rec(has, K), 3) for K in (1, 3, 5)])
    for s in ("AIHUB", "CCD", "MMAU", "NEXAR"):
        xs = [x for x in has if x["source"] == s]; print(f"  {s:6s} n {len(xs):3d} recall@1/3/5", [round(rec(xs, K), 3) for K in (1, 3, 5)])
    for lo, hi in ((0, .5), (.5, 1), (1, 1.5), (1.5, 99)):
        xs = [x for x in has if lo <= x["gap_s"] < hi]; print(f"  gap {lo}-{hi} n {len(xs):3d} recall@3 {rec(xs, 3):.3f}")
    fixes = [x for x in has if x["OB_D_hit"] and not x["E4_hit"]]; hurts = [x for x in has if not x["OB_D_hit"] and x["E4_hit"]]
    print(f"  OB_D fixes E4 (n {len(fixes)}) recall@3 {rec(fixes, 3):.3f} | OB_D hurts E4 (n {len(hurts)}) recall@3 {rec(hurts, 3):.3f}")
    def cond(name, pred):
        a = [x for x in has if pred(x)]; b = [x for x in has if not pred(x)]
        f = lambda xs, k: np.mean([x[k] for x in xs]) if xs else float("nan")
        print(f"  {name:32s} yes n {len(a):3d} OB_D {f(a,'OB_D_hit'):.3f} E4 {f(a,'E4_hit'):.3f} | no n {len(b):3d} OB_D {f(b,'OB_D_hit'):.3f} E4 {f(b,'E4_hit'):.3f}")
    print("conditional ENTRY accuracy (native):")
    cond("oracle in current top-3", lambda x: x["oracle_rank"] <= 3)
    cond("lane valid at oracle", lambda x: x["oracle_feat"] and x["oracle_feat"]["lane_valid"])
    cond("oracle track stable (persist>=.8)", lambda x: x["oracle_feat"] and x["oracle_feat"]["persist"] >= .8)
    print(f"  no oracle found: n {len(recs)-len(has)} OB_D {np.mean([x['OB_D_hit'] for x in recs if x['oracle'] is None]):.3f} E4 {np.mean([x['E4_hit'] for x in recs if x['oracle'] is None]):.3f}")


if __name__ == "__main__": main()
