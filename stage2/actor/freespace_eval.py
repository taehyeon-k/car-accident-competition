"""Deepening Exp 3 — does free-space evidence (cache_freespace) improve `evasion_space`? Nested CV on out-of-fold predictions.

Per clip and condition, window = native frames [predicted ENTRY - 5, predicted COLLISION] of that condition's v8+D5H prediction
(no GT used for the window). Clip features: window mean of the 16 per-frame values, their mean over the last 6 frames before the
predicted COLLISION, side-relative copies (free space / vehicles on the predicted entry side vs the opposite side), and the
ensemble's evasion / side logits. Models (logistic regression, standardised features; C chosen by an inner 4-fold CV on the
outer-training folds by macro F1; decision at 0.5):
  A ensemble only (current)    B recalibrated ensemble logit    C free-space only    D ensemble + free-space
Trained on native-condition features of the outer-training folds, applied to every condition of the held-out fold.
Usage: python -m stage2.actor.freespace_eval [ARM_KEY]
"""
from __future__ import annotations

import json, sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from stage2.long_context_v2_experiments import common as C

S = C.REPO / "stage2/generalization/results/entry_suite"; FS = C.REPO / "stage2/actor/cache_freespace"
FOLDS = C.REPO / "stage2/long_context_v2_experiments/folds"; CS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
logit = lambda p: float(np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4))))


def clip_feats(q, cache):
    x = cache[q["sample_id"]]; n = len(x)
    e = int(np.clip(q["entry_frame"] - 5, 0, n - 1)); c = int(np.clip(q["collision_frame"], e, n - 1))
    w = x[e:c + 1].mean(0); last = x[max(e, c - 5):c + 1].mean(0); right = q["_psr"] >= .5
    same = [w[3], w[5], w[10], w[13], last[3], last[10]] if right else [w[2], w[4], w[8], w[11], last[2], last[8]]
    opp = [w[2], w[4], w[8], w[11], last[2], last[8]] if right else [w[3], w[5], w[10], w[13], last[3], last[10]]
    return np.concatenate([w, last, same, opp]).astype(np.float32)


def fit(X, y, C_):
    sc = StandardScaler().fit(X); m = LogisticRegression(C=C_, max_iter=2000).fit(sc.transform(X), y)
    return lambda Z: m.predict(sc.transform(Z))


def choose_C(X, y, groups):
    best, bc = -1, CS[0]
    for C_ in CS:
        pred = np.zeros(len(y), int)
        for g in np.unique(groups):
            tr = groups != g; pred[~tr] = fit(X[tr], y[tr], C_)(X[~tr])
        f = C.macro_f1(list(y), list(pred))
        if f > best + 1e-9: best, bc = f, C_
    return bc


def main():
    arm = sys.argv[1] if len(sys.argv) > 1 else "E4_sa+E2_sa+XN4_sa+D5H@s012t"
    P = json.loads((S / f"{arm}.json").read_text())["preds"]
    fold_of = {r["sample_id"]: f for f in range(5) for r in C.rows(str(FOLDS / f"fold{f}_val.jsonl"))}
    cache = {q["sample_id"]: np.load(FS / f"{q['sample_id']}.npy").astype(np.float32) for q in P["k1"]}
    conds = [c for c in ("k1", "k2", "k3", "crop50", "crop25", "dacon") if c in P]
    idx = {c: {q["sample_id"]: q for q in P[c]} for c in conds}
    sids = [q["sample_id"] for q in P["k1"]]; y = np.array([idx["k1"][s]["evasion_gt"] for s in sids]); fo = np.array([fold_of[s] for s in sids])
    def design(c, kind):
        rows = []
        for s in sids:
            q = idx[c][s]; ens = [logit(q["_pev"]), logit(q["_psr"])]; fs = clip_feats(q, cache)
            rows.append({"B": ens[:1], "C": list(fs), "D": ens + list(fs)}[kind])
        return np.array(rows, np.float32)
    out = {}
    print(f"## {arm}: evasion macro F1 (score change vs A)\n\n| model | " + " | ".join(conds) + " | chosen C per fold |\n|---|" + "---|" * (len(conds) + 1))
    base = {c: C.metrics(P[c]) for c in conds}
    print("| A ensemble | " + " | ".join(f"{base[c]['evasion_f1']:.3f}" for c in conds) + " | – |")
    for kind, name in (("B", "B recalibrated"), ("C", "C free-space only"), ("D", "D ensemble + free-space")):
        Xtr = design("k1", kind); preds = {c: np.zeros(len(sids), int) for c in conds}; chosen = []
        Xc = {c: design(c, kind) for c in conds}
        for f in range(5):
            tr = fo != f; C_ = choose_C(Xtr[tr], y[tr], fo[tr]); chosen.append(C_); model = fit(Xtr[tr], y[tr], C_)
            for c in conds: preds[c][~tr] = model(Xc[c][~tr])
        cells = []
        for c in conds:
            new = [dict(idx[c][s], evasion_space=int(preds[c][i])) for i, s in enumerate(sids)]; nm = C.metrics(new)
            cells.append(f"{nm['evasion_f1']:.3f} ({nm['score'] - base[c]['score']:+.4f})"); out.setdefault(kind, {})[c] = nm["evasion_f1"]
        print(f"| {name} | " + " | ".join(cells) + f" | {chosen} |")
        if kind == "D":
            for s_ in ("AIHUB", "CCD", "MMAU", "NEXAR"):
                m_ = [i for i, s in enumerate(sids) if C.source(idx['k1'][s]) == s_]
                a_ = C.macro_f1([int(y[i]) for i in m_], [idx["k1"][sids[i]]["evasion_space"] for i in m_]); d_ = C.macro_f1([int(y[i]) for i in m_], [int(preds["k1"][i]) for i in m_])
                print(f"  {s_}: native evasion F1 A {a_:.3f} -> D {d_:.3f}")
    C.dump(C.REPO / "stage2/actor/results/freespace_eval.json", out)


if __name__ == "__main__": main()
