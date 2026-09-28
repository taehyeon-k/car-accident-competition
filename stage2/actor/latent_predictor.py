"""Stage2_experiments D3 — future latent prediction / transition surprise on the frozen representation (no large video model).

Latents (64-d, PCA, label-free): ACTOR = tight actor-crop DINOv3-S embedding (cache_actorcrop, PCA64 of actor_features);
GLOBAL = mean of the 70 DINO frame tokens (dense cache).
Predictor f (per latent, per CV fold, trained on that fold's TRAINING videos only, no labels): MLP([z_{t-3}, .., z_t]) -> (z_{t+1},
z_{t+2}, z_{t+4}) in steps of the (possibly strided) sequence; training sequences use native frames[o::s], s in {1, 2, 3}.
Surprise at step t for horizon h: log(1 + mean((z_t - f(z_{t-h-3..t-h})[h])^2)), rescaled in the cache x5 (actor) / x30 (global) to the
range of the other slot features (the object branch has no input normalisation). Per retained frame: actor s1, s2, s4, actor-available
flag, global s1, s2, s4 (7 values), appended to the a50 actor slot -> layout a57s (per fold: cache_actorfeat/a57s/fold{f}/...).
FPS-blind (steps are sampled frames). Usage: python -m stage2.actor.latent_predictor
"""
from __future__ import annotations

from multiprocessing import Pool

import numpy as np
import torch
from torch import nn

from stage2.long_context_v2_experiments import common as C
from .actor_features import OUT as FEAT, SPLITS, CROP, A, PCA64

GM = C.REPO / "stage2/actor/cache_gmean"; FOLDS = C.REPO / "stage2/long_context_v2_experiments/folds"
H = (1, 2, 4); CTX = 4


def gmean(sid):
    p = GM / f"{sid}.npy"
    if not p.exists():
        _, x = C.dense(sid); np.save(p, np.asarray(x, np.float32).mean(1).astype(np.float16))
    return np.load(p).astype(np.float32)


class Pred(nn.Module):
    def __init__(self, d=64):
        super().__init__(); self.net = nn.Sequential(nn.Linear(CTX * d, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, len(H) * d))
    def forward(self, x): return self.net(x.flatten(1)).view(len(x), len(H), -1)


def latents(sid, variant, p64, gpca):
    cr = np.load(CROP / f"{sid}.npz"); idx = cr[f"idx_{variant}"]; ok = idx >= 0
    za = np.zeros((len(idx), 64), np.float32)
    za[ok] = ((cr["tight"][idx[ok]].astype(np.float32) - p64["tight_mu"]) @ p64["tight_comp"].T) / p64["tight_scale"]
    zg = ((gmean(sid) - gpca["mu"]) @ gpca["comp"].T) / gpca["scale"]
    return za, ok, zg


def windows(z, ok, s, o):
    """(context [m, CTX, 64], targets [m, len(H), 64], target mask [m, len(H)]) over native frames[o::s]"""
    zs, oks = z[o::s], ok[o::s]; n = len(zs); X, Y, M = [], [], []
    for t in range(CTX - 1, n - 1):
        if not oks[t - CTX + 1:t + 1].all(): continue
        tg = [t + h for h in H]; m = [tt < n and oks[tt] for tt in tg]
        if not any(m): continue
        X.append(zs[t - CTX + 1:t + 1]); Y.append(np.stack([zs[tt] if tt < n else np.zeros(64, np.float32) for tt in tg])); M.append(m)
    return X, Y, M


def train_pred(seqs, dev, steps=3000):
    X, Y, M = [], [], []
    for z, ok in seqs:
        for s in (1, 2, 3):
            for o in range(s):
                x, y, m = windows(z, ok, s, o); X += x; Y += y; M += m
    X, Y, M = (torch.from_numpy(np.array(v, np.float32)).to(dev) for v in (X, Y, M))
    net = Pred().to(dev); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4); g = torch.Generator(device="cpu").manual_seed(0)
    for it in range(steps):
        b = torch.randint(len(X), (512,), generator=g).to(dev)
        loss = (((net(X[b]) - Y[b]) ** 2).mean(-1) * M[b]).sum() / M[b].sum().clamp_min(1)
        opt.zero_grad(); loss.backward(); opt.step()
    return net.eval(), float(loss), len(X)


@torch.no_grad()
def surprise(net, z, ok, k, dev):
    zs, oks = z[::k], ok[::k]; n = len(zs); out = np.zeros((n, len(H)), np.float32); avail = np.zeros(n, np.float32)
    ctx_ok = np.array([t >= CTX - 1 and oks[t - CTX + 1:t + 1].all() for t in range(n)])
    idx = np.flatnonzero(ctx_ok)
    if len(idx):
        P = net(torch.from_numpy(np.stack([zs[t - CTX + 1:t + 1] for t in idx])).to(dev)).cpu().numpy()   # predictions made at t
        for j, t in enumerate(idx):
            for hi, h in enumerate(H):
                if t + h < n and oks[t + h]: out[t + h, hi] = np.log1p(((zs[t + h] - P[j, hi]) ** 2).mean()); avail[t + h] = 1.0
    return out, avail


def main():
    dev = torch.device("cuda"); p64 = dict(np.load(PCA64)); sids = sorted(p.stem for p in (A / "gt").glob("*.npz"))
    GM.mkdir(exist_ok=True)
    with Pool(6) as p: p.map(gmean, sids)
    allg = np.concatenate([gmean(s)[::4] for s in sids]); mu = allg.mean(0); _, _, vt = np.linalg.svd(allg - mu, full_matrices=False)
    comp = vt[:64]; gpca = {"mu": mu, "comp": comp, "scale": ((allg - mu) @ comp.T).std(0) + 1e-6}
    lat = {s: latents(s, "gt", p64, gpca) for s in sids}
    for f in range(5):
        tr = [r["sample_id"] for r in C.rows(str(FOLDS / f"fold{f}_train.jsonl"))]
        na, la_, nw_a = train_pred([(lat[s][0], lat[s][1]) for s in tr], dev)
        ng, lg_, nw_g = train_pred([(lat[s][2], np.ones(len(lat[s][2]), bool)) for s in tr], dev)
        print(f"fold {f}: actor predictor loss {la_:.3f} ({nw_a} windows) | global predictor loss {lg_:.3f} ({nw_g} windows)", flush=True)
        for split, jobs in SPLITS.items():
            for variant, k, sub in jobs:
                d = FEAT / "a57s" / f"fold{f}" / split / sub; d.mkdir(parents=True, exist_ok=True)
                for s in sids:
                    base = FEAT / "a50" / split / sub / f"{s}.npy"
                    if not base.exists(): continue
                    za, ok, zg = latents(s, variant, p64, gpca) if variant != "gt" else lat[s]
                    sa, av = surprise(na, za, ok, k, dev); sg, _ = surprise(ng, zg, np.ones(len(zg), bool), k, dev)
                    np.save(d / f"{s}.npy", np.concatenate([np.load(base).astype(np.float32), 5 * sa, av[:, None], 30 * sg], 1).astype(np.float16))
    print("done")


if __name__ == "__main__": main()
