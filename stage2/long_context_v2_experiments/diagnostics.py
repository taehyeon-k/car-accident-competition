"""Phase 1 diagnostics.

  verify      dense cache == existing 192/383 caches; baseline val metrics reproduce exactly
  context     Experiment A: controlled NEXAR context-length benchmark (+ prediction drift)
  recall      Experiment B: Candidate Recall@K on the 70 val clips

Usage: python -m stage2.long_context_v2_experiments.diagnostics {verify,context,recall} [--models ...]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from . import common as C

OUT = C.REPO / "stage2/long_context_v2_experiments/results/diagnostics"
CONTEXTS = [128, 256, 512, 768, 1024, "full"]
KS = [1, 2, 4, 8, 16]


def verify(device):
    from stage2.temporal_pyramid_experiments.data import cached_features
    worst = 0.0
    for row in C.rows("val"):
        frames, feats = C.dense(row["sample_id"])
        pos = {int(f): i for i, f in enumerate(frames)}
        for cache in ("stage2/artifacts/spotting_features/fpsblind_192_7x10_349", "stage2/temporal_pyramid_experiments/cache_383"):
            cf, cx = cached_features(str(C.REPO / cache / f"{row['sample_id']}.npz"))
            d = np.abs(feats[[pos[int(f)] for f in cf]].astype(np.float32) - cx.astype(np.float32)).max()
            worst = max(worst, float(d))
    print(f"max |dense - cached| feature difference over val: {worst:.5f}")
    for name in C.BASELINES:
        model, sampling = C.load_named(name, device)
        preds = C.run(model, [C.make_item(r, sampling) for r in C.rows("val")], device)
        print(name, C.short_table(C.breakdown(preds)))


def crop_bounds(n, entry, collision, length):
    """Window of `length` frames containing [entry, collision] that keeps the event's relative position.

    Relative position r = event-midpoint / (n-1) is preserved, so at full length the crop is the clip
    itself, and shorter crops remove unrelated context proportionally from both sides."""
    if length == "full" or length >= n: return 0, n
    mid = (entry + collision) / 2
    r = mid / max(n - 1, 1)
    start = int(round(mid - r * (length - 1)))
    start = min(max(start, collision - length + 1, 0), entry, n - length)
    return start, start + length


def context(device, names):
    val = [r for r in C.rows("val") if C.source(r) == "NEXAR"]
    result, per_video = {}, {}
    for name in names:
        model, sampling = C.load_named(name, device)
        result[name], per_video[name] = {}, {}
        for L in CONTEXTS:
            items, info = [], []
            for row in val:
                frames, _ = C.dense(row["sample_id"])
                ei = int(np.searchsorted(frames, int(row["entry_frame"])))
                ci = int(np.searchsorted(frames, int(row["collision_frame"])))
                if L != "full" and ci - ei + 1 > L:  # event does not fit: skip rather than cut it
                    continue
                s, e = crop_bounds(len(frames), ei, ci, L)
                items.append(C.make_item(row, sampling, s, e)); info.append((s, e))
            preds = C.run(model, items, device)
            for p, (s, e) in zip(preds, info):
                p["crop"] = [s, e]
            result[name][str(L)] = C.metrics(preds)
            per_video[name][str(L)] = preds
        # drift: fraction of videos whose selected event moves by >10% of the *full* clip length between consecutive contexts
        drift = {}
        keys = [str(L) for L in CONTEXTS]
        for a, b in zip(keys, keys[1:]):
            pa = {p["sample_id"]: p for p in per_video[name][a]}; pb = {p["sample_id"]: p for p in per_video[name][b]}
            common_ids = sorted(set(pa) & set(pb))
            full_n = {r["sample_id"]: int(r["num_frames"]) for r in val}
            for e in ("entry", "collision"):
                moved = [abs(pa[i][f"{e}_frame"] - pb[i][f"{e}_frame"]) / full_n[i] > C.CATASTROPHIC for i in common_ids]
                drift[f"{a}->{b}:{e}_switch_rate"] = float(np.mean(moved)) if moved else None
                # switches that go from a hit to a miss (the harmful kind)
                lost = [C.hit(pa[i], e) and not C.hit(pb[i], e) for i in common_ids]
                drift[f"{a}->{b}:{e}_hit_lost"] = int(sum(lost))
                gained = [not C.hit(pa[i], e) and C.hit(pb[i], e) for i in common_ids]
                drift[f"{a}->{b}:{e}_hit_gained"] = int(sum(gained))
        result[name]["drift"] = drift
        print(f"\n== {name} ({sampling})")
        print(f"{'ctx':>5s} {'n':>3s} {'score':>6s} {'E':>5s} {'C':>5s} {'E_mae':>6s} {'C_mae':>6s} {'E_nmae':>6s} {'C_nmae':>6s} {'side':>5s} {'eva':>5s} {'Ecat':>5s} {'Ccat':>5s}")
        for L in CONTEXTS:
            m = result[name][str(L)]
            print(f"{str(L):>5s} {m['n']:3d} {m['score']:6.3f} {m['entry_acc']:5.2f} {m['collision_acc']:5.2f} {m['entry_mae_frames']:6.1f} "
                  f"{m['collision_mae_frames']:6.1f} {m['entry_norm_mae']:6.3f} {m['collision_norm_mae']:6.3f} {m['side_f1']:5.2f} {m['evasion_f1']:5.2f} "
                  f"{m['entry_catastrophic']:5.2f} {m['collision_catastrophic']:5.2f}")
        for k, v in drift.items():
            if "switch" in k: print("  ", k, v, "| lost", drift[k.replace("switch_rate", "hit_lost")], "gained", drift[k.replace("switch_rate", "hit_gained")])
    C.dump(OUT / "context_length.json", result)
    C.dump(OUT / "context_length_predictions.json", per_video)


def peaks(logits, radius, k):
    """Greedy NMS on a 1-D score sequence: top-k local maxima at least `radius` positions apart."""
    order = np.argsort(-np.asarray(logits)); taken = []
    for i in order:
        if all(abs(i - j) > radius for j in taken):
            taken.append(int(i))
            if len(taken) == k: break
    return taken


def recall_at_k(preds, max_k=16):
    """Per-video candidate lists and whether any of the top-K peaks is an official hit (0.3 s)."""
    fps = C.fps_table()
    records = []
    for p in preds:
        t = len(p["frames"]); radius = max(1, round(0.015 * (t - 1)))  # = training target width in positions
        rec = {"sample_id": p["sample_id"], "source": C.source(p), "n": p["num_available_frames"], "positions": t}
        for e in ("entry", "collision"):
            logits = np.asarray(p[f"{e}_logits"])
            cand = peaks(logits, radius, max_k)
            frames = [p["frames"][i] for i in cand]
            hits = [abs(f - p[f"{e}_gt"]) / fps[p["sample_id"]] <= .300001 for f in frames]
            probs = np.exp(logits - logits.max()); probs /= probs.sum()
            rec[f"{e}_candidates"] = [{"frame": int(f), "logit": float(logits[i]), "prob": float(probs[i]), "hit": bool(h)}
                                      for f, i, h in zip(frames, cand, hits)]
            rec[f"{e}_first_hit_rank"] = next((r + 1 for r, h in enumerate(hits) if h), None)
            # can a hit exist at all at this sampling density? (nearest sampled frame within tolerance)
            rec[f"{e}_reachable"] = bool(min(abs(f - p[f"{e}_gt"]) for f in p["frames"]) / fps[p["sample_id"]] <= .300001)
        records.append(rec)
    return records


def summarize_recall(records):
    def block(rs):
        out = {"n": len(rs)}
        for e in ("entry", "collision"):
            for k in KS:
                out[f"{e}_R@{k}"] = float(np.mean([r[f"{e}_first_hit_rank"] is not None and r[f"{e}_first_hit_rank"] <= k for r in rs])) if rs else None
            out[f"{e}_reachable"] = float(np.mean([r[f"{e}_reachable"] for r in rs])) if rs else None
        return out
    res = {"overall": block(records)}
    for s in sorted({r["source"] for r in records}):
        res[f"source:{s}"] = block([r for r in records if r["source"] == s])
    for _, _, b in C.BINS:
        sub = [r for r in records if C.frame_bin(r["n"]) == b]
        if sub: res[f"bin:{b}"] = block(sub)
    return res


def print_recall(name, res):
    print(f"\n== {name}")
    print(f"{'subset':16s} {'n':>3s} " + " ".join(f"E@{k:<3d}" for k in KS) + " E_reach | " + " ".join(f"C@{k:<3d}" for k in KS) + " C_reach")
    for key, m in res.items():
        print(f"{key:16s} {m['n']:3d} " + " ".join(f"{m[f'entry_R@{k}']:5.2f}" for k in KS) + f"  {m['entry_reachable']:5.2f} | "
              + " ".join(f"{m[f'collision_R@{k}']:5.2f}" for k in KS) + f"  {m['collision_reachable']:5.2f}")


def recall(device, names):
    summary, cands = {}, {}
    for name in names:
        model, sampling = C.load_named(name, device)
        preds = C.run(model, [C.make_item(r, sampling) for r in C.rows("val")], device, keep_logits=True)
        cands[name] = recall_at_k(preds)
        summary[name] = summarize_recall(cands[name])
        print_recall(name, summary[name])
    C.dump(OUT / "recall_at_k.json", summary)
    C.dump(OUT / "recall_candidates.json", cands)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=["verify", "context", "recall"])
    p.add_argument("--models", nargs="+", default=list(C.BASELINES))
    a = p.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda")
    {"verify": lambda: verify(device), "context": lambda: context(device, a.models), "recall": lambda: recall(device, a.models)}[a.task]()


if __name__ == "__main__": main()
