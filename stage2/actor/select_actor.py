"""Stage2_experiments D1 — collision-conditioned causal actor: select the vehicle most likely involved in COLLISION and keep its
identity backward through the clip (RF-DETR + ByteTrack tracks, cache_tracks; ego-lane corridor, cache_corridor).

Selection (no labels; c_hat = a COLLISION frame estimate): every vehicle track with detections in [c_hat - W, c_hat + 2] (W = 10 native
frames) is scored by the mean over those frames of
    log(area) / 5 + bottom y + ego-lane overlap + 0.5 clip(intrusion depth, -1, 1) + 0.5 centrality        (+ 0.5 x window coverage)
(the closest, lowest, most in-lane vehicle at the collision). Margin = best - second-best score.
Identity: ByteTrack fragments are linked backward from the selected track (and forward up to c_hat): a fragment ending <= LINK frames
before the chain's first frame is appended when its last box overlaps (IoU > .1) or is near (centre distance < .75 x size, size ratio
within [.4, 2.5]) the chain's first box. Gaps of <= LINK frames are linearly interpolated (present = 2).
Variants (c_hat source):
  gt          GT COLLISION (training clips only; DIAGNOSTIC / training target construction, never used for validation clips)
  e4v         E4_sa fold-k seed-0 out-of-fold native prediction (validation clips inside training runs; all 349 clips)
  v8_<cond>   v8 ensemble (E4_sa+E2_sa+XN4_sa) out-of-fold prediction under entry_suite condition <cond> (evaluation; clean clips)
Output cache_actor/<variant>/<sid>.npz: box [n, 4] (normalised), present [n] int8 (0 none, 1 detected, 2 interpolated),
tid [n] (source ByteTrack id), c_hat, margin, n_cand, chain. Frame indices only; no FPS.
Usage: python -m stage2.actor.select_actor
"""
from __future__ import annotations

import json
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

from stage2.long_context_v2_experiments import common as C
from stage2.objtrack.obj_crops import VEH
from stage2.objtrack.objlane_features import lane_at

TR = C.REPO / "stage2/objtrack/cache_tracks"; CORR = C.REPO / "stage2/objtrack/cache_corridor"
OUT = C.REPO / "stage2/actor/cache_actor"
SUITE = C.REPO / "stage2/generalization/results/entry_suite"
W, LINK = 10, 12
CONDS = ("k1", "k2", "k3", "crop50", "crop25")


def track_table(d):
    out = defaultdict(dict)
    for f, b, s, c, t in zip(d["frame"], d["box"], d["score"], d["cls"], d["track"]):
        if t >= 0 and c in VEH: out[int(t)][int(f)] = (np.asarray(b, np.float32), float(s))
    return out


def frame_score(b, corr, f):
    x1, y1, x2, y2 = [float(v) for v in b]; cx = (x1 + x2) / 2; w = x2 - x1
    s = np.log(max(w * (y2 - y1), 1e-5)) / 5 + y2 + 0.5 * (1 - 2 * abs(cx - .5))
    lane = lane_at(corr["left"][f].astype(np.float32), corr["right"][f].astype(np.float32), corr["row_y"], y2)
    if lane:
        Lb, Rb = lane; cc, hw = (Lb + Rb) / 2, (Rb - Lb) / 2
        depth = ((x2 - Lb) if cx < cc else (Rb - x1)) / hw
        s += max(0.0, min(x2, Rb) - max(x1, Lb)) / max(w, 1e-4) + 0.5 * float(np.clip(depth, -1, 1))
    return s


def select(tt, corr, c_hat, n):
    lo, hi = max(0, c_hat - W), min(n - 1, c_hat + 2); sc = {}
    for tid, fr in tt.items():
        fs = [f for f in range(lo, hi + 1) if f in fr]
        if fs: sc[tid] = float(np.mean([frame_score(fr[f][0], corr, f) for f in fs])) + 0.5 * len(fs) / (hi - lo + 1)
    if not sc: return None, 0.0, 0
    order = sorted(sc, key=lambda t: -sc[t])
    return order[0], (sc[order[0]] - sc[order[1]]) if len(order) > 1 else 1.0, len(order)


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / max(u, 1e-8)


def near(a, b):
    ca, cb = np.array([(a[0] + a[2]) / 2, (a[1] + a[3]) / 2]), np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
    sa, sb = max(a[2] - a[0], a[3] - a[1]), max(b[2] - b[0], b[3] - b[1]); r = sa / max(sb, 1e-6)
    return iou(a, b) > .1 or (np.linalg.norm(ca - cb) < .75 * max(sa, sb) and .4 <= r <= 2.5)


def link(tt, tid, c_hat):
    chain = [tid]; used = {tid}
    while True:  # backward
        first = min(min(tt[t]) for t in chain)
        fb = {f: tt[t][f][0] for t in chain for f in tt[t]}; b0 = fb[first]
        cands = []
        for t, fr in tt.items():
            if t in used: continue
            le = max(fr)
            if first - LINK <= le < first and near(fr[le][0], b0): cands.append((-iou(fr[le][0], b0), first - le, t))
        if not cands: break
        t = sorted(cands)[0][2]; chain.append(t); used.add(t)
    while True:  # forward, only up to the COLLISION estimate
        last = max(max(tt[t]) for t in chain)
        if last >= c_hat - 1: break
        fb = {f: tt[t][f][0] for t in chain for f in tt[t]}; b1 = fb[last]; cands = []
        for t, fr in tt.items():
            if t in used: continue
            fs = min(fr)
            if last < fs <= last + LINK and near(fr[fs][0], b1): cands.append((-iou(fr[fs][0], b1), fs - last, t))
        if not cands: break
        t = sorted(cands)[0][2]; chain.append(t); used.add(t)
    return chain


def actor_track(tt, chain, n):
    box = np.zeros((n, 4), np.float32); present = np.zeros(n, np.int8); tid = np.full(n, -1, np.int32)
    for t in chain:  # the first chain element (the collision track) wins on overlaps
        for f, (b, _) in tt[t].items():
            if present[f] == 0: box[f] = b; present[f] = 1; tid[f] = t
    idx = np.flatnonzero(present == 1)
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= LINK:
            for f in range(a + 1, b):
                w = (f - a) / (b - a); box[f] = (1 - w) * box[a] + w * box[b]; present[f] = 2; tid[f] = tid[a]
    return box, present, tid


def c_hats():
    """variant -> {sid: c_hat native frame}"""
    rows = C.rows("all"); out = {"gt": {r["sample_id"]: int(r["collision_frame"]) for r in rows}, "e4v": {}}
    for f in range(5):
        for p in json.loads((C.REPO / f"stage2/generalization/results/E4_sa/cv/fold{f}_seed0/predictions.json").read_text()):
            out["e4v"][p["sample_id"]] = int(p["collision_frame"])
    v8 = json.loads((SUITE / "E4_sa+E2_sa+XN4_sa.json").read_text())["preds"]
    for c in CONDS: out[f"v8_{c}"] = {p["sample_id"]: int(p["collision_frame"]) for p in v8[c]}
    return out


def one(args):
    sid, jobs = args
    d = np.load(TR / f"{sid}.npz"); n = int(d["nframes"]); corr = np.load(CORR / f"{sid}.npz"); tt = track_table(d); memo = {}
    for variant, c_hat in jobs:
        c_hat = int(min(max(c_hat, 0), n - 1)); target = OUT / variant / f"{sid}.npz"
        if c_hat not in memo:
            tid, margin, ncand = select(tt, corr, c_hat, n)
            if tid is None: memo[c_hat] = (np.zeros((n, 4), np.float32), np.zeros(n, np.int8), np.full(n, -1, np.int32), margin, ncand, [])
            else:
                chain = link(tt, tid, c_hat); memo[c_hat] = (*actor_track(tt, chain, n), margin, ncand, chain)
        box, present, tids, margin, ncand, chain = memo[c_hat]
        np.savez(target, box=box, present=present, tid=tids, c_hat=c_hat, margin=margin, n_cand=ncand, chain=np.array(chain, np.int32))


def main():
    ch = c_hats(); per = defaultdict(list)
    for v, m in ch.items():
        (OUT / v).mkdir(parents=True, exist_ok=True)
        for sid, c in m.items(): per[sid].append((v, c))
    with Pool(8) as p: p.map(one, sorted(per.items()), chunksize=4)
    print({v: len(m) for v, m in ch.items()})


if __name__ == "__main__": main()
