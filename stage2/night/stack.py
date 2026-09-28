"""Night campaign — learned temporal stacker over family ENTRY / COLLISION distributions (cheap at inference).

Inputs per sampled position: log p_ENTRY and log p_COLLISION of every listed family (generalization/ens_cache pickles, seeds 0-1
averaged, full positions) + their family means -> 1-D conv net (kernel 5, dilations 1/2/4, width 48, residual) -> ENTRY logits (and a
second net for COLLISION logits). Target: Gaussian (sigma 1 position) around the GT event position (as the direct loss).
Nested CV: for outer fold f the stacker is trained on the other four folds' clips (all conditions: native, 1/2, 1/3, crop50, crop25
as training examples) and applied to fold f. Decoding: constrained anchors on the stacked distributions; side / evasion = mean of the
families. Compared with the plain average of the same families. No FPS; positions only.
Usage: python -m stage2.night.stack FAM [FAM ...] [--epochs 60] [--coll-stack]
"""
from __future__ import annotations

import argparse, pickle

import numpy as np
import torch
from torch import nn

from stage2.long_context_v2_experiments import common as C
from stage2.spotting_experiments.objective import constrained_anchors
from stage2.generalization.ens_search import forecast_model
from stage2.generalization.robustness_analyze import excluded

CACHE = C.REPO / "stage2/generalization/results/ens_cache"; FOLDS = C.REPO / "stage2/long_context_v2_experiments/folds"
CONDS = ("k1_crop0.0", "k2_crop0.0", "k3_crop0.0", "k1_crop0.5", "k1_crop0.25")


class Net(nn.Module):
    def __init__(self, cin, w=48):
        super().__init__(); self.inp = nn.Conv1d(cin, w, 1)
        self.blocks = nn.ModuleList(nn.Sequential(nn.GELU(), nn.Conv1d(w, w, 5, padding=2 * d, dilation=d), nn.Dropout(0.2)) for d in (1, 2, 4))
        self.out = nn.Conv1d(w, 1, 1); nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.base = None  # residual: logits = log(family-mean probability) + correction (starts exactly at the plain average)

    def forward(self, x):  # x [B, C, T]; channel RES_CH holds log of the family-mean distribution of the stacked event
        h = self.inp(x)
        for b in self.blocks: h = h + b(h)
        return x[:, self.base] + self.out(h).squeeze(1)


def load(fams):
    D = {f: pickle.load(open(CACHE / f"{f}.pkl", "rb")) for f in fams}; EX = excluded(); out = {c: [] for c in CONDS}
    for c in CONDS:
        for i, rec in enumerate(D[fams[0]][c]):
            if rec["sid"] in EX: continue
            pe = np.stack([D[f][c][i]["views"][(1, 0)][0] for f in fams]); pc = np.stack([D[f][c][i]["views"][(1, 0)][1] for f in fams])
            side = np.mean([D[f][c][i]["views"][(1, 0)][2] for f in fams]); eva = np.mean([D[f][c][i]["views"][(1, 0)][3] for f in fams])
            fr = rec["frames"]; r = rec["row"]
            out[c].append({"sid": rec["sid"], "pe": pe, "pc": pc, "side": side, "eva": eva, "frames": fr, "row": r, "n_native": rec["n_native"],
                           "e": int(np.abs(fr - int(r["entry_frame"])).argmin()), "c": int(np.abs(fr - int(r["collision_frame"])).argmin())})
    return out


def feats(x):
    lpe, lpc = np.log(np.clip(x["pe"], 1e-8, 1)), np.log(np.clip(x["pc"], 1e-8, 1))
    me, mc = np.log(np.clip(x["pe"].mean(0, keepdims=True), 1e-8, 1)), np.log(np.clip(x["pc"].mean(0, keepdims=True), 1e-8, 1))
    return np.concatenate([lpe, lpc, me, mc]).astype(np.float32)  # last two channels: log of the family-mean ENTRY / COLLISION


def train(items, key, epochs, dev, seed=0, bs=32):
    """padded, masked mini-batches (positions beyond a clip's length get -inf logits)"""
    torch.manual_seed(seed); cin = feats(items[0]).shape[0]; net = Net(cin).to(dev); net.base = cin - 2 if key == "e" else cin - 1; opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-3)
    X = [feats(x) for x in items]; Y = np.array([x[key] for x in items]); L = np.array([x.shape[1] for x in X])
    for ep in range(epochs):
        net.train(); perm = np.random.default_rng(seed + ep).permutation(len(X))
        for s0 in range(0, len(perm), bs):
            idx = perm[s0:s0 + bs]; T = int(L[idx].max()); xb = np.zeros((len(idx), cin, T), np.float32); m = np.zeros((len(idx), T), bool)
            for j, i in enumerate(idx): xb[j, :, :L[i]] = X[i]; m[j, :L[i]] = True
            xb, m = torch.from_numpy(xb).to(dev), torch.from_numpy(m).to(dev); t = torch.arange(T, device=dev, dtype=torch.float32)[None]
            g = torch.exp(-0.5 * (t - torch.from_numpy(Y[idx]).to(dev)[:, None].float()) ** 2) * m; g = g / g.sum(-1, keepdim=True)
            lo = net(xb).masked_fill(~m, -1e4).log_softmax(-1); loss = -(g * lo).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return net.eval()


def decode(x, pe, pc):
    ei, ci = constrained_anchors(torch.tensor(pe)[None].clamp_min(1e-12).log(), torch.tensor(pc)[None].clamp_min(1e-12).log()); r = x["row"]; fr = x["frames"]
    return {"sample_id": x["sid"], "source_id": r["source_id"], "entry_frame": int(fr[int(ei[0])]), "collision_frame": int(fr[int(ci[0])]),
            "entry_side": int(x["side"] >= .5), "evasion_space": int(x["eva"] >= .5), "entry_gt": int(r["entry_frame"]), "collision_gt": int(r["collision_frame"]),
            "entry_side_gt": int(r["entry_side"] == "RIGHT"), "evasion_gt": int(r["evasion_space"]), "num_available_frames": x["n_native"]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("fams", nargs="+"); ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--coll-stack", action="store_true"); ap.add_argument("--bag", type=int, default=3); a = ap.parse_args()
    dev = torch.device("cuda"); data = load(a.fams); W = forecast_model()
    fold_of = {r["sample_id"]: f for f in range(5) for r in C.rows(str(FOLDS / f"fold{f}_val.jsonl"))}
    base = {c: [] for c in CONDS}; stk = {c: [] for c in CONDS}
    for f in range(5):
        tr = [x for c in CONDS for x in data[c] if fold_of[x["sid"]] != f]
        ne = [train(tr, "e", a.epochs, dev, s) for s in range(a.bag)]
        nc = [train(tr, "c", a.epochs, dev, 100 + s) for s in range(a.bag)] if a.coll_stack else None
        with torch.no_grad():
            for c in CONDS:
                for x in data[c]:
                    if fold_of[x["sid"]] != f: continue
                    pe0, pc0 = x["pe"].mean(0), x["pc"].mean(0); X = torch.from_numpy(feats(x))[None].to(dev)
                    pe = np.mean([n(X)[0].softmax(-1).cpu().numpy() for n in ne], 0)
                    pc = np.mean([n(X)[0].softmax(-1).cpu().numpy() for n in nc], 0) if nc else pc0
                    base[c].append(decode(x, pe0, pc0)); stk[c].append(decode(x, pe, pc))
        print("fold", f, "done", flush=True)
    for name, P in (("average", base), ("stacked", stk)):
        m = {c: C.metrics(P[c]) for c in CONDS}; fc = float(W @ [1, m["k3_crop0.0"]["score"], m["k1_crop0.25"]["entry_acc"]])
        print(f"{name:9s} score " + " / ".join(f"{m[c]['score']:.3f}" for c in CONDS) + " | ENTRY " + " / ".join(f"{m[c]['entry_acc']:.3f}" for c in CONDS)
              + " | COLL " + " / ".join(f"{m[c]['collision_acc']:.3f}" for c in CONDS) + f" | forecast {fc:.4f}", flush=True)


if __name__ == "__main__": main()
