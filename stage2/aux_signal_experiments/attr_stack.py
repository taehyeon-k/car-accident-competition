"""Event-window attribute stacker for side / evasion (no deep training; sklearn logistic regression).

Per clip, windows located at the ensemble's PREDICTED ENTRY / COLLISION sampled positions (e, c), W = 8 positions:
  geometry features (cache_geo, 83-d) and residual motion (cache_residual, 36-d, max over each sampled segment) averaged over
  [e-W, e), [e, c], (c, c+W]  + (between - before)  -> 4 x 119 dims, plus the ensemble's side / evasion probabilities (logit).
For CV fold k the model is fit only on the clips of the other folds (their OOF-located windows and labels), C chosen by an
inner 4-fold CV on those clips; applied to fold k. Positions / frame indices only, no FPS.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from stage2.long_context_v2_experiments import common as C
from .train import GEO, RESIDUAL, segments, residual_per_frame

W = 8


def clip_features(p):
    fr = np.asarray(p["frames"]); d = np.load(GEO / f"{p['sample_id']}.geo.npz")
    pos = np.searchsorted(d["frames"], fr); g = d["feats"][pos].astype(np.float32)
    frames_all, _ = C.dense(p["sample_id"]); abs_idx = np.searchsorted(frames_all, fr)
    r = segments(residual_per_frame(p["sample_id"]), abs_idx)[:, :36]
    x = np.concatenate([g, r], 1); T = len(x)
    e = int(np.argmin(np.abs(fr - p["entry_frame"]))); c = max(int(np.argmin(np.abs(fr - p["collision_frame"]))), e)
    before = x[max(e - W, 0):max(e, 1)].mean(0); between = x[e:c + 1].mean(0); after = x[c:min(c + W + 1, T)].mean(0)
    lg = lambda q: np.log(np.clip(q, 1e-4, 1 - 1e-4) / (1 - np.clip(q, 1e-4, 1 - 1e-4)))
    return np.concatenate([before, between, after, between - before, [lg(p["side_prob"]), lg(p["evasion_prob"])]])


def stack(folds):
    """folds: {k: [decoded preds with side_prob, evasion_prob, entry/collision_frame, frames, gts]} -> preds with new attributes."""
    X = {k: np.stack([clip_features(p) for p in ps]) for k, ps in folds.items()}
    out = []
    for k, ps in folds.items():
        Xtr = np.concatenate([X[j] for j in folds if j != k]); new = {}
        for attr, gt in (("side", "entry_side_gt"), ("evasion", "evasion_gt")):
            y = np.concatenate([[q[gt] for q in folds[j]] for j in folds if j != k])
            model = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                                 {"logisticregression__C": [0.003, 0.01, 0.03, 0.1, 0.3]}, cv=StratifiedKFold(4, shuffle=True, random_state=0),
                                 scoring="f1_macro")
            model.fit(Xtr, y); new[attr] = model.predict(X[k])
        for i, p in enumerate(ps):
            out.append({**p, "entry_side": int(new["side"][i]), "evasion_space": int(new["evasion"][i])})
    return out
