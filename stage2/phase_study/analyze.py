"""Aggregate phase-study runs.

CV numbers: pooled out-of-fold predictions over all 349 clips, one pool per CV seed; reported as mean +- std over CV seeds
(single-model protocol, as in reports/stage2_phase_loss_report.md). 'ens' = probability average over CV seeds per fold.
Fixed: 279/70 split, mean over seeds. Decoders (same checkpoints):
  D0 direct (constrained argmax of the direct heads; used for checkpoint selection)
  D1 transition (argmax of s_E / s_C from phase probabilities, entry <= collision)
  D2 local rerank: top-K direct ENTRY x top-K direct COLLISION pairs (e <= c), score = direct pair log-prob
     + w * (local ENTRY evidence + local COLLISION evidence), windows of W = 3 sampled positions on each side:
       ENTRY at i: mean log P(PRE) on [i-W, i-1] + mean log P(phase >= BETWEEN) on [i, i+W-1]
       COLL  at j: mean log P(phase <= BETWEEN) on [j-W, j-1] + mean log P(POST) on [j, j+W-1]
     w chosen per fold on the OTHER folds' OOF clips from {0.25, 0.5, 1, 2} (also reported at fixed w = 1).
  D3 existing global structured decoder (phase + transition, whole-clip sum) - control only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from stage2.long_context_v2_experiments import common as C
from stage2.long_context_v2_experiments.phase_loss import decode_structured, decode_transition
from stage2.spotting_experiments.train import macro_f1

R = C.REPO / "stage2/phase_study/results"
KEYS = [("overall", "all"), ("source:NEXAR", "NEXAR"), ("source:non-NEXAR", "nonNEX"), ("bin:>1000", ">1000")]
W = 3


# ------------------------------------------------------------------------------------------------ loading
def load_run(run, root=R):
    d = Path(root) / run
    fixed = [json.loads(p.read_text()) for p in sorted(d.glob("seed*/predictions.json"))]
    cv = {}
    for p in sorted((d / "cv").glob("fold*_seed*/predictions.json")):
        k = int(p.parent.name[4]); s = p.parent.name.split("_seed")[1]
        cv.setdefault(s, {})[k] = json.loads(p.read_text())
    cv = {s: v for s, v in cv.items() if len(v) == 5}
    metrics = [json.loads(p.read_text()) for p in sorted(d.glob("**/metrics.json"))]
    return fixed, cv, metrics


def logp(x): x = np.asarray(x, float); return x - np.logaddexp.reduce(x)


# ------------------------------------------------------------------------------------------------ decoders
def local_evidence(ph):
    """Per-position local ENTRY/COLLISION phase evidence (window W, no whole-clip accumulation). ph = [T,3] log-probs."""
    p = np.exp(ph); T = len(p)
    l_pre, l_ge_b = np.log(p[:, 0] + 1e-6), np.log(p[:, 1] + p[:, 2] + 1e-6)
    l_le_b, l_post = np.log(p[:, 0] + p[:, 1] + 1e-6), np.log(p[:, 2] + 1e-6)
    def wmean(a, lo, hi):
        lo, hi = max(lo, 0), min(hi, T)
        return a[lo:hi].mean() if hi > lo else 0.0  # before the clip start: virtual PRE (log 1 = 0)
    ev_e = np.array([wmean(l_pre, i - W, i) + wmean(l_ge_b, i, i + W) for i in range(T)])
    ev_c = np.array([wmean(l_le_b, j - W, j) + wmean(l_post, j, j + W) for j in range(T)])
    return ev_e, ev_c


def decode_pair(p, name, w=1.0, k=8):
    el, cl = logp(p["entry_logits"]), logp(p["collision_logits"])
    if name == "D0":
        pair = el[:, None] + cl[None]; pair = np.where(np.triu(np.ones_like(pair, bool)), pair, -np.inf)
        return np.unravel_index(np.argmax(pair), pair.shape)
    ph = np.asarray(p["phase_logp"], float)
    if name == "D1": return decode_transition(torch.tensor(ph))
    if name == "D3": return decode_structured(torch.tensor(ph), w_phase=1, w_tr=1)
    ev_e, ev_c = p.setdefault("_ev", local_evidence(ph))
    ei = np.argsort(-el)[:k]; ci = np.argsort(-cl)[:k]
    best, arg = -np.inf, None
    for i in ei:
        for j in ci:
            if i <= j:
                s = el[i] + cl[j] + w * (ev_e[i] + ev_c[j])
                if s > best: best, arg = s, (i, j)
    if arg is None:  # no ordered pair among the candidates: fall back to direct decoding
        return decode_pair(p, "D0")
    return arg


def redecode(preds, name, **kw):
    out = []
    for p in preds:
        i, j = decode_pair(p, name, **kw)
        q = {k: v for k, v in p.items() if k not in ("entry_logits", "collision_logits", "phase_logp", "frames", "_ev")}
        q["entry_frame"], q["collision_frame"] = int(p["frames"][int(i)]), int(p["frames"][int(j)])
        out.append(q)
    return out


def d2_cv(folds, k):
    """D2 on pooled OOF of one CV seed: weight chosen per fold on the other folds."""
    ws = (0.25, 0.5, 1.0, 2.0)
    cache = {(f, w): redecode(folds[f], "D2", w=w, k=k) for f in folds for w in ws}
    out, chosen = [], []
    for f in folds:
        w = max(ws, key=lambda w: C.metrics([q for g in folds if g != f for q in cache[(g, w)]])["score"])
        chosen.append(w); out += cache[(f, w)]
    return out, chosen


# ------------------------------------------------------------------------------------------------ metrics
def phase_metrics(preds):
    if not preds or "phase_logp" not in preds[0]: return {}
    ys, ps, te, tc = [], [], [], []
    for p in preds:
        T = len(p["frames"]); t = np.arange(T)
        y = (t >= p["entry_index_gt"]).astype(int) + (t >= p["collision_index_gt"]).astype(int)
        ph = np.asarray(p["phase_logp"]); ys.append(y); ps.append(ph.argmax(1))
        i, j = decode_transition(torch.tensor(ph))
        te.append(abs(i - p["entry_index_gt"])); tc.append(abs(j - p["collision_index_gt"]))
    y, pr = np.concatenate(ys), np.concatenate(ps)
    f1 = np.mean([2 * ((pr == c) & (y == c)).sum() / max((pr == c).sum() + (y == c).sum(), 1) for c in range(3)])
    return {"phase_acc": float((y == pr).mean()), "phase_macro_f1": float(f1),
            "tr_err_entry": float(np.mean(te)), "tr_err_collision": float(np.mean(tc)),
            "tr_within1_entry": float(np.mean(np.array(te) <= 1)), "tr_within1_collision": float(np.mean(np.array(tc) <= 1))}


def flat(preds):
    b = C.breakdown(preds); out = {}
    for key, short in KEYS:
        m = b.get(key, {})
        if not m.get("n"): continue
        for f in ("score", "entry_acc", "collision_acc", "side_f1", "evasion_f1", "entry_norm_mae", "collision_norm_mae",
                  "entry_catastrophic", "collision_catastrophic"):
            out[f"{short}.{f}"] = m[f]
    return out


def mean_std(dicts):
    keys = dicts[0].keys()
    return {k: (float(np.mean([d[k] for d in dicts])), float(np.std([d[k] for d in dicts]))) for k in keys}


def seed_ensemble(cv):
    """Probability-average the CV seeds per fold (direct heads + attributes)."""
    seeds = sorted(cv); out = []
    for k in range(5):
        for items in zip(*[cv[s][k] for s in seeds]):
            p = dict(items[0])
            for e in ("entry", "collision"):
                p[f"{e}_logits"] = np.log(np.mean([np.exp(logp(x[f"{e}_logits"])) for x in items], 0) + 1e-12).tolist()
            p["side_prob_right"] = float(np.mean([x["side_prob_right"] for x in items]))
            p["evasion_prob"] = float(np.mean([x["evasion_prob"] for x in items]))
            if "phase_logp" in p:
                p["phase_logp"] = np.log(np.mean([np.exp(x["phase_logp"]) for x in items], 0) + 1e-12).tolist()
            p["entry_side"] = int(p["side_prob_right"] >= .5); p["evasion_space"] = int(p["evasion_prob"] >= .5)
            out.append(p)
    return redecode(out, "D0")


def summarize(run, decoders=True, root=R):
    fixed, cv, metrics = load_run(run, root)
    res = {"run": run, "n_fixed": len(fixed), "n_cv_seeds": len(cv)}
    if fixed: res["fixed"] = mean_std([flat(p) for p in fixed])
    if not cv: return res
    pooled = {s: [p for k in range(5) for p in cv[s][k]] for s in cv}
    res["cv"] = mean_std([{**flat(p), **phase_metrics(p)} for p in pooled.values()])
    res["cv_per_fold"] = [float(np.mean([C.metrics(cv[s][k])["score"] for s in cv])) for k in range(5)]
    res["cv_seed_scores"] = {s: C.metrics(p)["score"] for s, p in pooled.items()}
    res["cv_seed_ensemble"] = flat(seed_ensemble(cv))
    res["train_val_gap"] = float(np.mean([m["train_official_at_best"] - m["breakdown"]["overall"]["score"] for m in metrics]))
    res["best_epochs"] = float(np.mean([m["best_epoch"] for m in metrics]))
    if decoders and "phase_logp" in pooled[next(iter(pooled))][0]:
        dec = {}
        for name in ("D1", "D3"):
            dec[name] = mean_std([flat(redecode(p, name)) for p in pooled.values()])
        for k in (4, 8):
            dec[f"D2_K{k}_w1"] = mean_std([flat(redecode(p, "D2", w=1.0, k=k)) for p in pooled.values()])
            runs = [d2_cv(cv[s], k) for s in cv]
            dec[f"D2_K{k}_cvw"] = mean_std([flat(r[0]) for r in runs]); dec[f"D2_K{k}_cvw"]["_w"] = [r[1] for r in runs]
        res["decoders"] = dec
    return res


# ------------------------------------------------------------------------------------------------ paired comparison
def clip_arrays(preds):
    preds = sorted(preds, key=lambda p: p["sample_id"])
    fps = C.fps_table()
    e = np.array([abs(p["entry_frame"] - p["entry_gt"]) / fps[p["sample_id"]] <= .300001 for p in preds], float)
    c = np.array([abs(p["collision_frame"] - p["collision_gt"]) / fps[p["sample_id"]] <= .300001 for p in preds], float)
    return {"ids": [p["sample_id"] for p in preds], "e": e, "c": c,
            "sp": np.array([p["entry_side"] for p in preds]), "sg": np.array([p["entry_side_gt"] for p in preds]),
            "vp": np.array([p["evasion_space"] for p in preds]), "vg": np.array([p["evasion_gt"] for p in preds]),
            "long": np.array([C.source(p) == "NEXAR" for p in preds])}


def f1_binary_macro(pred, gt):
    out = []
    for c in (0, 1):
        tp = ((pred == c) & (gt == c)).sum(-1); den = (pred == c).sum(-1) + (gt == c).sum(-1)
        out.append(np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0))
    return (out[0] + out[1]) / 2


def score_idx(a, idx):
    return (.35 * a["e"][idx].mean(-1) + .35 * a["c"][idx].mean(-1) + .15 * f1_binary_macro(a["sp"][idx], a["sg"][idx])
            + .15 * f1_binary_macro(a["vp"][idx], a["vg"][idx]))


def paired(run, control, n_boot=2000, root=R, control_root=R):
    _, cv_a, _ = load_run(run, root); _, cv_b, _ = load_run(control, control_root)
    A = [clip_arrays([p for k in range(5) for p in cv_a[s][k]]) for s in sorted(cv_a)]
    B = [clip_arrays([p for k in range(5) for p in cv_b[s][k]]) for s in sorted(cv_b)]
    n = len(A[0]["e"]); rng = np.random.default_rng(0); idx = rng.integers(0, n, (n_boot, n))
    diff = np.mean([score_idx(a, idx) for a in A], 0) - np.mean([score_idx(b, idx) for b in B], 0)
    full = np.arange(n)[None]
    point = float(np.mean([score_idx(a, full) for a in A]) - np.mean([score_idx(b, full) for b in B]))
    # per-clip event hits (entry + collision), averaged over seeds
    ha = np.mean([a["e"] + a["c"] for a in A], 0); hb = np.mean([b["e"] + b["c"] for b in B], 0); d = ha - hb
    order = np.argsort(-np.abs(d))
    top = float(np.abs(d[order[:10]]).sum() / max(np.abs(d).sum(), 1e-9))
    return {"delta": point, "ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
            "p_gt0": float((diff > 0).mean()), "clips_gain": int((d > 0.25).sum()), "clips_loss": int((d < -0.25).sum()),
            "net_hits": float(d.sum()), "share_of_change_in_top10_clips": top,
            "delta_entry_hits": float(np.mean([a["e"].sum() for a in A]) - np.mean([b["e"].sum() for b in B])),
            "delta_collision_hits": float(np.mean([a["c"].sum() for a in A]) - np.mean([b["c"].sum() for b in B]))}


def fold_wins(res, ctrl):
    return int(sum(a > b for a, b in zip(res["cv_per_fold"], ctrl["cv_per_fold"])))


# ------------------------------------------------------------------------------------------------ table
def fmt(ms, key, digits=3):
    if key not in ms: return "—"
    m, s = ms[key]; return f"{m:.{digits}f}±{s:.{digits}f}"


def table(results, control=None):
    hdr = ["run", "CV all", "NEXAR", ">1000", "E acc", "C acc", "side F1", "eva F1", "E nMAE", "C nMAE", "E cat", "C cat",
           ">1000 C cat", "phase acc", "phase F1", "tr err E/C", "seed-ens CV all", "fixed all", "folds won", "Δ vs ctrl [95% CI]", "gap"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    ctl = next((x for x in results if x["run"] == control), None) if control else None
    for r in results:
        cv = r["cv"]; v = lambda k: f"{cv[k][0]:.3f}" if k in cv else "—"
        pr = r.get("paired")
        cells = [r["run"], fmt(cv, "all.score"), fmt(cv, "NEXAR.score"), fmt(cv, ">1000.score"), v("all.entry_acc"), v("all.collision_acc"),
                 v("all.side_f1"), v("all.evasion_f1"), v("all.entry_norm_mae"), v("all.collision_norm_mae"), v("all.entry_catastrophic"),
                 v("all.collision_catastrophic"), v(">1000.collision_catastrophic"), v("phase_acc"), v("phase_macro_f1"),
                 f"{cv['tr_err_entry'][0]:.1f}/{cv['tr_err_collision'][0]:.1f}" if "tr_err_entry" in cv else "—",
                 f"{r['cv_seed_ensemble']['all.score']:.3f}", fmt(r["fixed"], "all.score") if "fixed" in r else "—",
                 f"{fold_wins(r, ctl)}/5" if ctl is not None and ctl is not r else "",
                 f"{pr['delta']:+.3f} [{pr['ci95'][0]:+.3f},{pr['ci95'][1]:+.3f}]" if pr else "", f"{r['train_val_gap']:.3f}"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+"); ap.add_argument("--control", default=None)
    ap.add_argument("--no-decoders", action="store_true"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    results = []
    for run in a.runs:
        r = summarize(run, decoders=not a.no_decoders)
        if a.control and run != a.control: r["paired"] = paired(run, a.control)
        results.append(r)
    print(table(results, a.control))
    for r in results:
        if r.get("paired"): print(r["run"], "vs", a.control, json.dumps(r["paired"]))
    for r in results:
        if "decoders" in r:
            print(f"\n{r['run']} decoders (CV, mean±std over seeds):")
            base = r["cv"]
            print(f"  D0 direct       all {fmt(base, 'all.score')} NEXAR {fmt(base, 'NEXAR.score')} >1000 {fmt(base, '>1000.score')} "
                  f"Ecat {base['all.entry_catastrophic'][0]:.3f} Ccat {base['all.collision_catastrophic'][0]:.3f}")
            for name, m in r["decoders"].items():
                print(f"  {name:15s} all {fmt(m, 'all.score')} NEXAR {fmt(m, 'NEXAR.score')} >1000 {fmt(m, '>1000.score')} "
                      f"Ecat {m['all.entry_catastrophic'][0]:.3f} Ccat {m['all.collision_catastrophic'][0]:.3f}" + (f" w={m['_w']}" if '_w' in m else ""))
    if a.out: C.dump(a.out, results)


if __name__ == "__main__": main()
