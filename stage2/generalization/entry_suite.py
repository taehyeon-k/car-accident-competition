"""Experiment 0 of stage2/Stage2_Entry_Robustness_Experiments.md — ENTRY robustness evaluation suite.

For CV arms (families joined with '+', seeds 0-2 by default; duplicate-clean validation clips) under five conditions — native, 1/2 and
1/3 frame rate (motion recomputed), and random 50 % / 25 % windows containing ENTRY..COLLISION (robust_eval.item) — every member's
probabilities are computed once; the ensemble and each single seed (all families of that seed) are decoded from them (plain decoding:
probability average + constrained anchors). Per-clip predictions are saved (results/entry_suite/<arm>.json).
Reported: score / ENTRY / COLLISION per condition; ENTRY hit and signed timing error (median, mean; s) per ENTRY->COLLISION gap bin
(pooled over the three rates); crop ENTRY hit and predicted-vs-true ENTRY position slope inside the window; source-wise ENTRY;
per-seed values (variance); statistical-prior baselines (timing only, no video evidence):
  PRIOR_POS   ENTRY at the training-median relative position of ENTRY in the clip / window
  PRIOR_GAPR  ENTRY = the reference model's COLLISION - training-median gap as a fraction of the clip / window length
FPS is used only by the metric and for gap binning. `--entry-key` decodes another ENTRY output (e.g. an ENTRY branch alone).
"""
from __future__ import annotations

import argparse, json
from collections import defaultdict

import numpy as np
import torch

from stage2.long_context_v2_experiments import common as C
from stage2.aux_signal_experiments.model import load as load_model
from stage2.spotting_experiments.objective import constrained_anchors
from .robust_eval import item, root, FOLDS
from stage2.aux_signal_experiments.train import segments
from .clean_eval import EXCL

RES = C.REPO / "stage2/generalization/results/entry_suite"
CONDS = {"k1": (1, 0.0), "k2": (2, 0.0), "k3": (3, 0.0), "crop50": (1, 0.5), "crop25": (1, 0.25)}
DACON_STRIDE = {"CCD": 1, "AIHUB": 2}; DACON_LEN = 50  # --dacon: + DACON-like view (robustness_profile.item_dacon window rule)
GAPS = [(0, .5), (.5, 1), (1, 1.5), (1.5, 2.5), (2.5, 99)]
TTA_VIEWS = []  # --tta: head-level multi-rate views (s, o): positions[o::s], s in 2, 3 (only when T >= 3 s)
TTA_W = []; TTA_G = [0]
ENTRY_FAMS = set()  # --entry-fams A,B: ENTRY averaged over these families only (COLLISION / attributes over all members)
FUSE_EXPERT = False; EXP_A = []  # --fuse-expert A,..: pre-collision ENTRY expert members (truncate_near) run on the prefix before the reference's predicted COLLISION; ENTRY = (1-a) others + a expert
CUT = {}  # --cut-at KEY: {cond: {sid: predicted COLLISION frame}} -> inputs end before it; ENTRY = argmax inside, COLLISION = the reference's
PHYS = 0.0  # --phys w: lightweight physical decoder weight (0 = plain decoding)


def members(run, fold, seeds, dev):
    out = []
    for r in run.split("+"):
        for s in seeds:
            c = root(r) / r / "cv" / f"fold{fold}_seed{s}" / "checkpoint.pt"
            if not c.exists(): continue
            cfg = torch.load(c, map_location="cpu", weights_only=False)["config"]
            kind = {"both": "both", "global": "global"}.get(cfg.get("motion", "none"), None)
            if cfg.get("objmotion"): kind = "both_objfeat" if cfg.get("obj_cache") else "both_obj"
            mdl = load_model(c, dev); mdl.pce_expert = bool(cfg.get("truncate_near")) and FUSE_EXPERT; mdl.fam = r
            out.append((mdl, kind, cfg.get("feats_dir", ""), s, eval_cache(cfg)))
    return out


def eval_cache(cfg):
    """evaluation object cache of a member: actor caches <layout>[/fold{f}]/train -> .../eval; otherwise OBJ_CACHE (None)"""
    oc = cfg.get("obj_cache") or ""
    if "cache_flowgrid" in oc: return oc   # per-frame feature cache already in the k-view layout (night campaign flow grid)
    if "cache_actorfeat" not in oc: return None
    parts = oc.rstrip("/").split("/"); parts[-1] = "eval"
    if parts[-2].startswith("fold"): parts[-2] = "fold{fold}"
    return "/".join(parts)


@torch.inference_mode()
def member_probs(ms, it, dev, entry_key):
    v = torch.ones(1, len(it["frames"]), dtype=torch.bool, device=dev); x = it["x"][None].to(dev); xs = {}; out = []
    for m, kind, fd, s, oc in ms:
        if oc and kind == "both_objfeat": kind = oc   # member-specific object cache (item computed per cache in run_arm)
        xm = x
        if fd:
            if fd not in xs: xs[fd] = torch.from_numpy(np.ascontiguousarray(np.load(f"{fd}/{it['sid']}.npy", mmap_mode="r")[it["abs"]]))[None].to(dev)
            xm = xs[fd]
        src = it["_oc"][kind] if (kind and kind not in it) else it
        mo = (src["both_objfeat"] if kind not in it else it[kind]) if kind else None
        if getattr(m, "pce_expert", False):  # prefix of the sampled positions before the predicted COLLISION (>= 4), zero mass after it
            T = xm.shape[1]; tc = max(int((np.asarray(it["frames"]) < it["_fuse_c"]).sum()), min(4, T))
            oe = m(xm[:, :tc], v[:, :tc], motion=mo[None, :tc].to(dev)) if kind else m(xm[:, :tc], v[:, :tc])
            pe_ = torch.zeros(T, device=dev); pe_[:tc] = oe[entry_key if entry_key in oe else "entry_logits"][0].float().softmax(-1)
            out.append((s, pe_, None, None, None, None, None, "expert")); continue
        o = m(xm, v, motion=mo[None].to(dev)) if kind else m(xm, v)
        ek = entry_key if entry_key in o else "entry_logits"
        views = None
        if TTA_VIEWS:  # same sampled frames, thinned; motion / object features re-segmented over the view (as the runtime would)
            T = xm.shape[1]; pev, pcv = [], []
            for st_, of_ in TTA_VIEWS:
                if T < 3 * st_: continue
                sub = np.arange(of_, T, st_); ps_ = it["_pos"][sub]
                g = segments(it["_m28"], ps_); r_ = segments(it["_res"], ps_)
                if kind == "global": mv = g
                elif kind == "both": mv = np.concatenate([g, r_], 1)
                elif kind: mv = np.concatenate([g, r_, segments(src["_obf"], ps_)], 1)
                vv = torch.ones(1, len(sub), dtype=torch.bool, device=dev)
                ov = m(xm[:, sub], vv, motion=torch.from_numpy(mv)[None].to(dev)) if kind else m(xm[:, sub], vv)
                pev.append(spread(ov[ek][0].float().softmax(-1).cpu().numpy(), T, st_, of_)); pcv.append(spread(ov["collision_logits"][0].float().softmax(-1).cpu().numpy(), T, st_, of_))
            if pev: views = (torch.from_numpy(np.mean(pev, 0)).to(dev), torch.from_numpy(np.mean(pcv, 0)).to(dev))
        out.append((s, o[ek][0].float().softmax(-1), o["collision_logits"][0].float().softmax(-1),
                    float(o["side_logits"][0].float().softmax(-1)[1]), float(o["evasion_logits"].float().sigmoid().reshape(-1)[0]),
                    o["obj_state_logits"][0].float().softmax(-1) if "obj_state_logits" in o else None, views, None, getattr(m, "fam", "")))
    return out


def spread(p, T, s, o):
    """view distribution over positions o, o+s, ... -> full positions (ens_cache.spread): position t takes the mass of the view position
    whose segment (previous view position, this view position] contains t, shared equally"""
    full = np.zeros(T, np.float32); idx = np.arange(o, T, s); lo = 0
    for j, a in enumerate(idx):
        hi = a if j < len(idx) - 1 else T - 1
        full[lo:hi + 1] += p[j] / (hi - lo + 1); lo = hi + 1
    return full / full.sum()


def decode(parts, it, r, w=0.0, gate=0, alpha=0.0):
    if gate and len(it["frames"]) < gate: w = 0.0   # length gate: views only for clips with >= gate sampled positions
    experts = [p for p in parts if len(p) > 7 and p[7] == "expert"]; parts = [p for p in parts if not (len(p) > 7 and p[7] == "expert")]
    mix = lambda p, j: (1 - w) * p[j] + w * p[6][j - 1] if (w and len(p) > 6 and p[6] is not None) else p[j]
    ep = [p for p in parts if len(p) > 8 and p[8] in ENTRY_FAMS] if ENTRY_FAMS else parts   # --entry-fams: ENTRY from a family subset
    pe = torch.stack([mix(p, 1) for p in (ep or parts)]).mean(0); pc = torch.stack([mix(p, 2) for p in parts]).mean(0)
    if experts and alpha: pe = (1 - alpha) * pe + alpha * torch.stack([p[1] for p in experts]).mean(0)
    le = pe.log()
    if PHYS and all(len(p) > 5 and p[5] is not None for p in parts):  # Stage2_experiments lightweight physical decoder (no gap / position prior)
        st = torch.stack([p[5] for p in parts]).mean(0); T = len(pe); m = 3
        ps = pe.clone(); ps[1:-1] = (pe[:-2] + 2 * pe[1:-1] + pe[2:]) / 4                      # no isolated one-position spikes
        csum = lambda x: torch.cat([x.new_zeros(1), x.cumsum(0)])
        cb, ca = csum(st[:, 0]), csum(st[:, 2]); t = torch.arange(T, device=pe.device)
        lo, hi = (t - m).clamp_min(0), (t + m).clamp_max(T - 1)
        before = torch.where(t > 0, (cb[t] - cb[lo]) / (t - lo).clamp_min(1), torch.full_like(pe, .5))
        after = torch.where(t < T - 1, (ca[hi + 1] - ca[t + 1]) / (hi - t).clamp_min(1), torch.full_like(pe, .5))
        le = ps.clamp_min(1e-12).log() + PHYS * (before + after - 1)                            # AFTER persists, BEFORE precedes
    ei, ci = constrained_anchors(le[None], pc.log()[None]); fr = it["frames"]
    if it.get("_cut_c") is not None: ei = le.argmax()[None]
    return {"sample_id": r["sample_id"], "source_id": r["source_id"], "entry_frame": int(fr[int(ei[0])]),
            "collision_frame": int(it["_cut_c"]) if it.get("_cut_c") is not None else int(fr[int(ci[0])]),
            "entry_side": int(np.mean([p[3] for p in parts]) >= .5), "evasion_space": int(np.mean([p[4] for p in parts]) >= .5),
            "entry_gt": int(r["entry_frame"]), "collision_gt": int(r["collision_frame"]), "entry_side_gt": int(r["entry_side"] == "RIGHT"),
            "evasion_gt": int(r["evasion_space"]), "num_available_frames": it["n_native"], "_lo": int(fr[0]), "_hi": int(fr[-1]), "_n": len(fr),
            "_npk": n_peaks(pe.cpu().numpy()), "_psr": float(np.mean([p[3] for p in parts])), "_pev": float(np.mean([p[4] for p in parts]))}


def n_peaks(p, rel=0.25):
    """ENTRY distribution modes: local maxima >= rel x the global maximum, merged when closer than 3 positions (multi-peak diagnostic)"""
    m = p.max(); pk = [t for t in range(len(p)) if p[t] >= rel * m and (t == 0 or p[t] >= p[t - 1]) and (t == len(p) - 1 or p[t] > p[t + 1])]
    out = []
    for t in pk:
        if not out or t - out[-1] >= 3: out.append(t)
    return len(out)


def run_arm(run, seeds, dev, entry_key="entry_logits"):
    P = {c: {"ens": [], **{f"s{s}": [] for s in seeds}} for c in CONDS}
    from . import robust_eval
    for f in range(5):
        ms = members(run, f, seeds, dev); robust_eval.FOLD_HINT = f
        for r in C.rows(str(FOLDS / f"fold{f}_val.jsonl")):
            if r["sample_id"] in EXCL: continue
            for cond, (k, cr) in CONDS.items():
                fl = None
                if cond == "dacon": k, cr, fl = DACON_STRIDE.get(r["source_id"].split(":")[0], 3), 0.0, DACON_LEN
                ef = CUT[cond][r["sample_id"]] if CUT else None
                it = item(r, k, cr, fixed_len=fl, end_frame=ef); it["_oc"] = {m_[4]: item(r, k, cr, oc=m_[4], fixed_len=fl, end_frame=ef) for m_ in ms if m_[4]}
                it["_cut_c"] = ef
                if FUSE_EXPERT: it["_fuse_c"] = FUSE_C[cond][r["sample_id"]]
                parts = member_probs(ms, it, dev, entry_key)
                P[cond]["ens"].append(decode(parts, it, r))
                for a_ in EXP_A: P[cond].setdefault(f"a{a_}", []).append(decode(parts, it, r, alpha=a_))
                for s_ in seeds:
                    for a_ in EXP_A:
                        sp = [p for p in parts if p[0] == s_]
                        if sp: P[cond].setdefault(f"a{a_}s{s_}", []).append(decode(sp, it, r, alpha=a_))
                for w in TTA_W:
                    for g in TTA_G: P[cond].setdefault(f"w{w}" + (f"g{g}" if g else ""), []).append(decode(parts, it, r, w, g))
                for s in seeds:
                    sp = [p for p in parts if p[0] == s]
                    if sp: P[cond][f"s{s}"].append(decode(sp, it, r))
    return P


def metrics(P, fps):
    hit = lambda p, e: abs(p[f"{e}_frame"] - p[f"{e}_gt"]) / fps[p["sample_id"]] <= .300001
    gap = lambda p: (p["collision_gt"] - p["entry_gt"]) / fps[p["sample_id"]]
    out = {}
    for c in CONDS:
        m = C.metrics(P[c]); out[c] = {k: round(m[k], 4) for k in ("score", "entry_acc", "collision_acc", "side_f1", "evasion_f1")}
    pooled = P["k1"] + P["k2"] + P["k3"]
    out["gap"] = {}
    for lo, hi in GAPS:
        ps = [p for p in pooled if lo <= gap(p) < hi]; err = np.array([(p["entry_frame"] - p["entry_gt"]) / fps[p["sample_id"]] for p in ps])
        out["gap"][f"{lo}-{hi}"] = {"n": len(ps), "entry_acc": round(float(np.mean([hit(p, "entry") for p in ps])), 3),
                                    "coll_acc": round(float(np.mean([hit(p, "collision") for p in ps])), 3),
                                    "bias_median_s": round(float(np.median(err)), 3), "bias_mean_s": round(float(err.mean()), 3)}
    lg = [p for p in pooled if gap(p) >= 1.5]
    out["long_gap_bias_median_s"] = round(float(np.median([(p["entry_frame"] - p["entry_gt"]) / fps[p["sample_id"]] for p in lg])), 3)
    gl = np.array([np.log(max(gap(p), .05)) for p in pooled]); hl = np.array([hit(p, "entry") for p in pooled], float)
    out["entry_vs_log_gap_slope"] = round(float(np.polyfit(gl, hl, 1)[0]), 3)
    for c in ("crop50", "crop25"):
        rel = lambda x, p: (x - p["_lo"]) / max(p["_hi"] - p["_lo"], 1)
        t = np.array([rel(p["entry_gt"], p) for p in P[c]]); y = np.array([rel(p["entry_frame"], p) for p in P[c]])
        out[c]["slope"] = round(float(np.polyfit(t, y, 1)[0]), 3); out[c]["mean_pred_pos"] = round(float(y.mean()), 3); out[c]["mean_true_pos"] = round(float(t.mean()), 3)
    out["source"] = {s: {c: round(float(np.mean([hit(p, "entry") for p in P[c] if C.source(p) == s])), 3) for c in ("k1", "k3", "crop25")}
                     for s in ("AIHUB", "CCD", "MMAU", "NEXAR")}
    return out


def prior_baselines(P_ref, fps):
    """timing-only baselines fitted per fold on the training labels (median ENTRY relative position, median gap fraction);
    COLLISION taken from the reference arm's ensemble prediction under the same condition."""
    fold_of = {}
    for f in range(5):
        for r in C.rows(str(FOLDS / f"fold{f}_val.jsonl")): fold_of[r["sample_id"]] = f
    stats = {}
    for f in range(5):
        tr = C.rows(str(FOLDS / f"fold{f}_train.jsonl"))
        rel = [int(r["entry_frame"]) / max(int(r["num_frames"]) - 1, 1) for r in tr]
        gfrac = [(int(r["collision_frame"]) - int(r["entry_frame"])) / max(int(r["num_frames"]) - 1, 1) for r in tr]
        stats[f] = (float(np.median(rel)), float(np.median(gfrac)))
    out = {}
    for name in ("PRIOR_POS", "PRIOR_GAPR"):
        Q = {}
        for c in CONDS:
            Q[c] = {"ens": []}
            for p in P_ref[c]["ens"]:
                rp, gf = stats[fold_of[p["sample_id"]]]; span = p["_hi"] - p["_lo"]; q = dict(p)
                e = p["_lo"] + rp * span if name == "PRIOR_POS" else p["collision_frame"] - gf * span
                q["entry_frame"] = int(round(min(max(e, p["_lo"]), q["collision_frame"]))); Q[c]["ens"].append(q)
        out[name] = metrics({c: Q[c]["ens"] for c in CONDS}, fps)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--entry-key", default="entry_logits"); ap.add_argument("--tag", default=""); ap.add_argument("--priors", action="store_true")
    ap.add_argument("--tta", default="", help="head-level multi-rate TTA: comma list of view weights w (p = (1-w) full + w mean(views))")
    ap.add_argument("--tta-gates", default="0", help="comma list of minimum sampled positions for the TTA views (0 = always)")
    ap.add_argument("--fuse-expert", default="", help="comma list of ENTRY weights a for pre-collision expert members (needs --fuse-ref)")
    ap.add_argument("--fuse-ref", default="", help="suite key whose per-condition predicted COLLISION cuts the expert's input")
    ap.add_argument("--cut-at", default="", help="suite result key whose per-condition predicted COLLISION ends every input (pre-collision ENTRY expert)")
    ap.add_argument("--entry-fams", default="", help="comma list of families whose ENTRY distributions are averaged (default all)")
    ap.add_argument("--dacon", action="store_true", help="also evaluate the DACON-like view (50-frame window, source stride)")
    ap.add_argument("--phys", type=float, default=0.0, help="lightweight physical decoder weight on the object-state persistence terms")
    a = ap.parse_args(); dev = torch.device("cuda"); fps = C.fps_table(); RES.mkdir(parents=True, exist_ok=True)
    global PHYS; PHYS = a.phys
    if a.dacon: CONDS["dacon"] = (1, 0.0)
    global ENTRY_FAMS; ENTRY_FAMS = set(x for x in a.entry_fams.split(",") if x)
    if a.fuse_expert:
        global FUSE_EXPERT, EXP_A, FUSE_C; FUSE_EXPERT = True; EXP_A = [float(x) for x in a.fuse_expert.split(",")]
        ref = json.loads((RES / f"{a.fuse_ref}.json").read_text())["preds"]; FUSE_C = {c: {q["sample_id"]: int(q["collision_frame"]) for q in ref[c]} for c in CONDS}
    if a.cut_at:
        global CUT; ref = json.loads((RES / f"{a.cut_at}.json").read_text())["preds"]
        CUT = {c: {q["sample_id"]: int(q["collision_frame"]) for q in ref[c]} for c in CONDS}
    if a.tta:
        global TTA_VIEWS, TTA_W; TTA_VIEWS = [(2, 0), (2, 1), (3, 0), (3, 1), (3, 2)]; TTA_W = [float(x) for x in a.tta.split(",")]
        global TTA_G; TTA_G = [int(x) for x in a.tta_gates.split(",")]
    for run in a.runs:
        key = run + (f"@{a.tag}" if a.tag else "")
        P = run_arm(run, a.seeds, dev, a.entry_key)
        res = {"ensemble": metrics({c: P[c]["ens"] for c in CONDS}, fps),
               "per_seed": {f"s{s}": metrics({c: P[c][f"s{s}"] for c in CONDS}, fps) for s in a.seeds}}
        if a.priors: res["priors"] = prior_baselines(P, fps)
        C.dump(RES / f"{key}.json", {"metrics": res, "preds": {c: P[c]["ens"] for c in CONDS}})
        for vk in [f"a{a_}" for a_ in EXP_A] + [f"w{w}" + (f"g{g}" if g else "") for w in TTA_W for g in TTA_G]:
            mw = metrics({c: P[c][vk] for c in CONDS}, fps); C.dump(RES / f"{key}@tta{vk[1:]}.json", {"metrics": {"ensemble": mw}, "preds": {c: P[c][vk] for c in CONDS}})
            print(f"{key + f'@tta{vk[1:]}':36s} S " + "/".join(f"{mw[c]['score']:.3f}" for c in CONDS) + " | E " + "/".join(f"{mw[c]['entry_acc']:.3f}" for c in CONDS)
                  + " | C " + "/".join(f"{mw[c]['collision_acc']:.3f}" for c in CONDS) + f" | bias {mw['long_gap_bias_median_s']:+.2f}s", flush=True)
        e = res["ensemble"]; g = e["gap"]
        seeds_lg = [res["per_seed"][f"s{s}"]["gap"]["1.5-2.5"]["entry_acc"] for s in a.seeds]
        print(f"{key:36s} S {e['k1']['score']:.3f}/{e['k2']['score']:.3f}/{e['k3']['score']:.3f} | E {e['k1']['entry_acc']:.3f}/{e['k3']['entry_acc']:.3f} "
              f"| gapE " + "/".join(f"{g[b]['entry_acc']:.2f}" for b in g) + f" | 1.5-2.5 seeds {seeds_lg} | bias {e['long_gap_bias_median_s']:+.2f}s "
              f"| crop25 E {e['crop25']['entry_acc']:.3f} slope {e['crop25']['slope']:.2f} | crop50 E {e['crop50']['entry_acc']:.3f} slope {e['crop50']['slope']:.2f}", flush=True)
        if a.priors:
            for pn, pm in res["priors"].items():
                g = pm["gap"]; print(f"   {pn:12s} E {pm['k1']['entry_acc']:.3f}/{pm['k3']['entry_acc']:.3f} | gapE " + "/".join(f"{g[b]['entry_acc']:.2f}" for b in g)
                                     + f" | bias {pm['long_gap_bias_median_s']:+.2f}s | crop25 E {pm['crop25']['entry_acc']:.3f} slope {pm['crop25']['slope']:.2f}", flush=True)


if __name__ == "__main__": main()


@torch.inference_mode()
def gapood(run, seeds, dev):
    """gap-OOD: models trained on short/medium gaps (L split) scored on long-gap clips, and trained on medium/long gaps (S split) scored
    on short-gap clips; native and 1/3 rate; ensemble + per seed + a timing prior fitted on the same training split."""
    fps = C.fps_table(); G = C.REPO / "stage2/generalization/gapood"; out = {}
    for sp in ("L", "S"):
        ms = []
        for r_ in run.split("+"):
            for s in seeds:
                c = root(r_) / r_ / f"{sp}_seed{s}" / "checkpoint.pt"
                if not c.exists(): continue
                cfg = torch.load(c, map_location="cpu", weights_only=False)["config"]
                kind = {"both": "both", "global": "global"}.get(cfg.get("motion", "none"), None)
                if cfg.get("objmotion"): kind = "both_objfeat" if cfg.get("obj_cache") else "both_obj"
                ms.append((load_model(c, dev), kind, cfg.get("feats_dir", ""), s, eval_cache(cfg)))
        tr = C.rows(str(G / f"{sp}_train.jsonl"))
        gfrac = float(np.median([(int(r["collision_frame"]) - int(r["entry_frame"])) / max(int(r["num_frames"]) - 1, 1) for r in tr]))
        P = {k: {"ens": [], "prior": [], **{f"s{s}": [] for s in seeds}} for k in ("k1", "k3")}
        for r in C.rows(str(G / f"{sp}_val.jsonl")):
            if r["sample_id"] in EXCL: continue
            for key, k in (("k1", 1), ("k3", 3)):
                it = item(r, k); parts = member_probs(ms, it, dev, "entry_logits"); p = decode(parts, it, r); P[key]["ens"].append(p)
                q = dict(p); q["entry_frame"] = int(round(max(p["_lo"], p["collision_frame"] - gfrac * (p["_hi"] - p["_lo"])))); P[key]["prior"].append(q)
                for s in seeds:
                    sp_ = [x for x in parts if x[0] == s]
                    if sp_: P[key][f"s{s}"].append(decode(sp_, it, r))
        hit = lambda p: abs(p["entry_frame"] - p["entry_gt"]) / fps[p["sample_id"]] <= .300001
        bias = lambda ps: float(np.median([(p["entry_frame"] - p["entry_gt"]) / fps[p["sample_id"]] for p in ps]))
        out.setdefault("_preds", {})[sp] = {key: P[key]["ens"] for key in P}
        out[sp] = {key: {m: {"entry_acc": round(float(np.mean([hit(p) for p in P[key][m]])), 3), "bias_s": round(bias(P[key][m]), 3),
                             "score": round(C.metrics(P[key][m])["score"], 4)} for m in P[key]} for key in P}
    return out


def gapood_main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); a = ap.parse_args()
    dev = torch.device("cuda"); RES.mkdir(parents=True, exist_ok=True)
    for run in a.runs:
        o = gapood(run, a.seeds, dev); C.dump(RES / f"gapood_{run}.json", o)
        for sp in ("L", "S"):
            for key in ("k1", "k3"):
                d = o[sp][key]  # (per-clip ensemble predictions saved under o["_preds"]); sd = [d[f"s{s}"]["entry_acc"] for s in a.seeds if f"s{s}" in d]
                print(f"{run:24s} gap-OOD {sp} {key}: ENTRY ens {d['ens']['entry_acc']:.3f} (seeds {sd}) bias {d['ens']['bias_s']:+.2f}s score {d['ens']['score']:.3f} "
                      f"| prior {d['prior']['entry_acc']:.3f} bias {d['prior']['bias_s']:+.2f}s", flush=True)
