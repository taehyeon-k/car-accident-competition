"""Leave-one-source-out summary. For each run and held-out source: mean +- std over seeds of the single-model score, the seed-ensemble
score (probability average), ENTRY / COLLISION accuracy; across sources: mean and worst source (generalization = the worst source
improves, not only the mean). Paired per-source deltas vs a control run.
Usage: python -m stage2.generalization.loso_analyze RUN [RUN ...] --control LOSO_E4
"""
from __future__ import annotations

import argparse, json

import numpy as np

from stage2.long_context_v2_experiments import common as C
from stage2.phase_study.analyze import redecode, logp

R = C.REPO / "stage2/generalization/results"
SOURCES = ("AIHUB", "CCD", "MMAU", "NEXAR")


def load(run):
    out = {}
    for src in SOURCES:
        seeds = [json.loads(p.read_text()) for p in sorted((R / run).glob(f"{src}_seed*/predictions.json"))]
        if seeds: out[src] = seeds
    return out


def ensemble(seed_preds):
    base = []
    for items in zip(*seed_preds):
        p = dict(items[0])
        for e in ("entry", "collision"):
            p[f"{e}_logits"] = np.log(np.mean([np.exp(logp(x[f"{e}_logits"])) for x in items], 0) + 1e-12).tolist()
        p["entry_side"] = int(np.mean([x["side_prob_right"] for x in items]) >= .5)
        p["evasion_space"] = int(np.mean([x["evasion_prob"] for x in items]) >= .5)
        base.append(p)
    return redecode(base, "D0")


def summarize(run):
    d = load(run); res = {}
    for src, seeds in d.items():
        single = [C.metrics(s) for s in seeds]; ens = C.metrics(ensemble(seeds)) if len(seeds) > 1 else single[0]
        res[src] = {"n_seeds": len(seeds), "score": float(np.mean([m["score"] for m in single])), "std": float(np.std([m["score"] for m in single])),
                    "E": float(np.mean([m["entry_acc"] for m in single])), "C": float(np.mean([m["collision_acc"] for m in single])),
                    "side": float(np.mean([m["side_f1"] for m in single])), "eva": float(np.mean([m["evasion_f1"] for m in single])),
                    "ens": ens["score"], "ens_E": ens["entry_acc"], "ens_C": ens["collision_acc"]}
    full = [s for s in SOURCES if s in res]
    if len(full) == 4:
        res["_mean"] = float(np.mean([res[s]["score"] for s in SOURCES])); res["_worst"] = float(min(res[s]["score"] for s in SOURCES))
        res["_ens_mean"] = float(np.mean([res[s]["ens"] for s in SOURCES])); res["_ens_worst"] = float(min(res[s]["ens"] for s in SOURCES))
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--control", default=None); ap.add_argument("--out", default=None)
    a = ap.parse_args(); allres = {r: summarize(r) for r in a.runs}
    print("| run | " + " | ".join(f"{s} (E/C) [ens]" for s in SOURCES) + " | mean | worst | ens mean | ens worst |")
    print("|---|" + "---|" * (len(SOURCES) + 4))
    for run, res in allres.items():
        cells = [f"{res[s]['score']:.3f}±{res[s]['std']:.3f} ({res[s]['E']:.2f}/{res[s]['C']:.2f}) [{res[s]['ens']:.3f}]" if s in res else "—" for s in SOURCES]
        tail = [f"{res[k]:.3f}" if k in res else "—" for k in ("_mean", "_worst", "_ens_mean", "_ens_worst")]
        print(f"| {run} | " + " | ".join(cells) + " | " + " | ".join(tail) + " |")
    if a.control and a.control in allres:
        c = allres[a.control]
        for run, res in allres.items():
            if run == a.control or "_mean" not in res or "_mean" not in c: continue
            d = {s: res[s]["score"] - c[s]["score"] for s in SOURCES}
            print(f"{run} vs {a.control}: " + " ".join(f"{s} {v:+.3f}" for s, v in d.items()) +
                  f" | mean {res['_mean'] - c['_mean']:+.3f} worst {res['_worst'] - c['_worst']:+.3f} | sources improved {sum(v > 0 for v in d.values())}/4")
    if a.out: C.dump(a.out, allres)


if __name__ == "__main__": main()
