"""Stage2_experiments D1 — learned collision-actor selector (replaces the hand-set select_actor.select score).

Candidates: vehicle tracks with detections in [c_hat - W, c_hat + 2]. Label-free candidate features at c_hat (window = those frames;
history = the track's frames <= c_hat):
  window means: log area / 5, bottom y, ego-lane overlap, clipped intrusion depth, centrality, detector confidence, lane-defined rate;
  coverage of the window; growth (log area, last window frame - first track frame in [c_hat - 3W, c_hat]);
  history: largest 5-frame intrusion-depth rise, minimum depth (was it outside the lane), log track length.
Model: conditional logit (softmax over the candidates of a clip, 14 weights, L2) fitted per fold on the training clips of that fold with
target = the Exp A oracle ENTRY vehicle (diag_selection.json; GT-derived, training targets only) and c_hat = GT COLLISION.
Every clip's actor is chosen by the ranker of the fold in which it is a VALIDATION clip (out-of-fold), for every c_hat variant.
Output: cache_actor_rank/<variant>/<sid>.npz (same layout as select_actor). Usage: python -m stage2.actor.ranker
"""
from __future__ import annotations

import json
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import torch

from stage2.long_context_v2_experiments import common as C
from stage2.long_context_v2_experiments.common import REPO
from stage2.objtrack.objlane_features import lane_at
from .select_actor import track_table, link, actor_track, c_hats, W, TR, CORR

OUT = REPO / "stage2/actor/cache_actor_rank"
FOLDS = REPO / "stage2/long_context_v2_experiments/folds"
NF = 14


def geom(b, corr, f):
    x1, y1, x2, y2 = [float(v) for v in b]; cx = (x1 + x2) / 2; w = x2 - x1
    lane = lane_at(corr["left"][f].astype(np.float32), corr["right"][f].astype(np.float32), corr["row_y"], y2)
    ov = dep = 0.0; ok = 0.0
    if lane:
        Lb, Rb = lane; cc, hw = (Lb + Rb) / 2, (Rb - Lb) / 2; ok = 1.0
        dep = float(np.clip(((x2 - Lb) if cx < cc else (Rb - x1)) / hw, -3, 3)); ov = max(0.0, min(x2, Rb) - max(x1, Lb)) / max(w, 1e-4)
    return np.log(max(w * (y2 - y1), 1e-5)) / 5, y2, ov, dep, 1 - 2 * abs(cx - .5), ok


def cand_features(tt, corr, c_hat, n):
    lo, hi = max(0, c_hat - W), min(n - 1, c_hat + 2); tids, X = [], []
    for tid, fr in tt.items():
        fs = [f for f in range(lo, hi + 1) if f in fr]
        if not fs: continue
        g = np.array([geom(fr[f][0], corr, f) for f in fs]); conf = np.mean([fr[f][1] for f in fs])
        hist = sorted(f for f in fr if f <= hi); gh = {f: geom(fr[f][0], corr, f) for f in hist}
        early = [f for f in hist if f >= c_hat - 3 * W]; growth = float(np.clip(gh[fs[-1]][0] - gh[early[0]][0], -1, 1)) * 5 if early else 0.0
        dep = {f: gh[f][3] for f in hist if gh[f][5]}
        rise = max([dep[f] - dep[f - 5] for f in dep if f - 5 in dep], default=0.0)
        X.append([g[:, 0].mean(), g[:, 1].mean(), g[:, 2].mean(), np.clip(g[:, 3], -1, 1).mean(), g[:, 4].mean(), conf, g[:, 5].mean(),
                  len(fs) / (hi - lo + 1), growth, float(np.clip(rise, -2, 3)), float(np.clip(min(dep.values(), default=0.0), -3, 1)),
                  np.log1p(len(hist)) / 5, 1.0 if dep else 0.0, float(np.clip(max(dep.values(), default=0.0), -3, 3))])
        tids.append(tid)
    return tids, np.array(X, np.float32).reshape(-1, NF)


def clip_cands(args):
    sid, chs = args
    d = np.load(TR / f"{sid}.npz"); n = int(d["nframes"]); corr = np.load(CORR / f"{sid}.npz"); tt = track_table(d)
    return sid, {v: cand_features(tt, corr, int(min(max(c, 0), n - 1)), n) for v, c in chs}


def fit(data):
    """data: list of (X [m, NF], target index). conditional logit with feature standardisation."""
    allx = np.concatenate([x for x, _ in data]); mu, sd = allx.mean(0), allx.std(0) + 1e-6
    w = torch.zeros(NF, requires_grad=True); opt = torch.optim.LBFGS([w], max_iter=200, line_search_fn="strong_wolfe")
    Xs = [torch.from_numpy((x - mu) / sd) for x, _ in data]; ys = [y for _, y in data]
    def closure():
        opt.zero_grad(); loss = sum(-torch.log_softmax(x @ w, 0)[y] for x, y in zip(Xs, ys)) / len(Xs) + 1e-2 * (w ** 2).sum(); loss.backward(); return loss
    opt.step(closure)
    return w.detach().numpy(), mu, sd


def write(args):
    sid, jobs = args
    d = np.load(TR / f"{sid}.npz"); n = int(d["nframes"]); tt = track_table(d)
    for variant, c_hat, tids, scores in jobs:
        c_hat = int(min(max(c_hat, 0), n - 1))
        if not tids:
            np.savez(OUT / variant / f"{sid}.npz", box=np.zeros((n, 4), np.float32), present=np.zeros(n, np.int8), tid=np.full(n, -1, np.int32),
                     c_hat=c_hat, margin=0.0, n_cand=0, chain=np.zeros(0, np.int32)); continue
        o = np.argsort(-scores); tid = tids[o[0]]; margin = float(scores[o[0]] - scores[o[1]]) if len(o) > 1 else 5.0
        chain = link(tt, tid, c_hat); box, present, tt_ = actor_track(tt, chain, n)
        np.savez(OUT / variant / f"{sid}.npz", box=box, present=present, tid=tt_, c_hat=c_hat, margin=margin, n_cand=len(tids), chain=np.array(chain, np.int32))


def main():
    ch = c_hats(); per = defaultdict(list)
    for v, m in ch.items():
        (OUT / v).mkdir(parents=True, exist_ok=True)
        for sid, c in m.items(): per[sid].append((v, c))
    with Pool(8) as p: cands = dict(p.map(clip_cands, sorted(per.items()), chunksize=4))
    orc = {x["sid"]: x["oracle"] for x in json.loads((REPO / "stage2/generalization/results/diag_selection.json").read_text()) if x["oracle"] is not None}
    fold_of = {r["sample_id"]: f for f in range(5) for r in C.rows(str(FOLDS / f"fold{f}_val.jsonl"))}
    jobs = defaultdict(list); acc = []
    for f in range(5):
        tr = [s for s in orc if fold_of.get(s) != f]; data = []
        for s in tr:
            tids, X = cands[s]["gt"]
            if orc[s] in tids: data.append((X, tids.index(orc[s])))
        w, mu, sd = fit(data)
        print(f"fold {f}: fitted on {len(data)} clips; weights", np.round(w, 2).tolist(), flush=True)
        for s, fo in fold_of.items():
            if fo != f: continue
            for v, _ in per[s]:
                tids, X = cands[s][v]; sc = ((X - mu) / sd) @ w if len(tids) else np.zeros(0)
                jobs[s].append((v, ch[v][s], tids, sc))
                if v == "gt" and s in orc and tids: acc.append(tids[int(np.argmax(sc))] == orc[s])
    print(f"out-of-fold top-1 agreement with the oracle vehicle at GT COLLISION: {np.mean(acc):.3f} (n {len(acc)})")
    with Pool(8) as p: p.map(write, sorted(jobs.items()), chunksize=4)


if __name__ == "__main__": main()
