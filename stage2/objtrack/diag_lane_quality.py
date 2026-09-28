"""Narrowing campaign, Experiment C (diagnostic) — ego-lane corridor quality per clip and its relation to OB_D success.
Per clip (cache_corridor; near field = lower 45 % of the label rows): fraction of near-field rows with a defined corridor, fraction
of frames with >= 30 % defined near-field rows, lane-width stability (median over rows of the temporal CV of the width), mean
frame-to-frame boundary jump (normalised by the lane width), all over the whole clip and over ENTRY ± 1 s. No labels except for the
ENTRY window location; no FPS except for that window. Output results/diag_lane_quality.json."""
import json
import numpy as np
from stage2.long_context_v2_experiments import common as C
from stage2.generalization.clean_eval import EXCL

CORR = C.REPO / "stage2/objtrack/cache_corridor"


def quality(L, R):
    near = slice(int(L.shape[1] * 0.55), L.shape[1]); L, R = L[:, near].astype(np.float32), R[:, near].astype(np.float32)
    ok = np.isfinite(L) & np.isfinite(R) & (R - L > 0.01); w = np.where(ok, R - L, np.nan)
    with np.errstate(all="ignore"):
        cvw = np.nanmedian(np.nanstd(w, 0) / np.nanmean(w, 0))
        jump = np.nanmean(np.abs(np.diff(np.where(ok, (L + R) / 2, np.nan), axis=0)) / np.nanmean(w))
    return {"defined_rows": float(ok.mean()), "frames_ok": float((ok.mean(1) >= 0.3).mean()),
            "width_cv": float(cvw) if np.isfinite(cvw) else None, "centre_jump": float(jump) if np.isfinite(jump) else None}


def main():
    fps = C.fps_table(); suite = C.REPO / "stage2/generalization/results/entry_suite"
    pr = {k: {q["sample_id"]: q for q in json.loads((suite / f"{f}.json").read_text())["preds"]["k1"]} for k, f in (("OB_D", "OB_D@s6"), ("E4", "E4_sa@s6"))}
    hit = lambda q: abs(q["entry_frame"] - q["entry_gt"]) / fps[q["sample_id"]] <= .300001
    out = []
    for r in C.rows("all"):
        sid = r["sample_id"]
        if sid in EXCL: continue
        d = np.load(CORR / f"{sid}.npz"); e = int(r["entry_frame"]); h = int(round(fps[sid]))
        q_all = quality(d["left"], d["right"]); q_e = quality(d["left"][max(0, e - h):e + h + 1], d["right"][max(0, e - h):e + h + 1])
        out.append({"sid": sid, "source": C.source(r), "all": q_all, "entry": q_e, "OB_D_hit": hit(pr["OB_D"][sid]), "E4_hit": hit(pr["E4"][sid])})
    C.dump(C.REPO / "stage2/generalization/results/diag_lane_quality.json", out)
    med = lambda xs, k, w="entry": float(np.nanmedian([x[w][k] if x[w][k] is not None else np.nan for x in xs]))
    print("source   n   defined_rows frames_ok width_cv centre_jump   (ENTRY +-1 s)")
    for s in ("AIHUB", "CCD", "MMAU", "NEXAR"):
        xs = [x for x in out if x["source"] == s]
        print(f"{s:6s} {len(xs):4d}   {med(xs,'defined_rows'):.2f}        {med(xs,'frames_ok'):.2f}     {med(xs,'width_cv'):.3f}    {med(xs,'centre_jump'):.3f}")
    for name, xs in (("OB_D fixes E4", [x for x in out if x["OB_D_hit"] and not x["E4_hit"]]), ("OB_D hurts E4", [x for x in out if not x["OB_D_hit"] and x["E4_hit"]]),
                     ("both hit", [x for x in out if x["OB_D_hit"] and x["E4_hit"]]), ("both miss", [x for x in out if not x["OB_D_hit"] and not x["E4_hit"]])):
        print(f"{name:14s} n {len(xs):3d}   defined {med(xs,'defined_rows'):.2f} frames_ok {med(xs,'frames_ok'):.2f} width_cv {med(xs,'width_cv'):.3f} jump {med(xs,'centre_jump'):.3f}")
    # conditional accuracy by lane quality tercile (frames_ok around ENTRY)
    v = np.array([x["entry"]["frames_ok"] for x in out]); t1, t2 = np.quantile(v, [1/3, 2/3])
    for lab, sel in (("low", v <= t1), ("mid", (v > t1) & (v <= t2)), ("high", v > t2)):
        xs = [x for x, s_ in zip(out, sel) if s_]
        print(f"lane quality {lab:4s} (frames_ok {'<=' if lab=='low' else '>'}{t1 if lab!='high' else t2:.2f}) n {len(xs):3d}  OB_D {np.mean([x['OB_D_hit'] for x in xs]):.3f}  E4 {np.mean([x['E4_hit'] for x in xs]):.3f}")


if __name__ == "__main__": main()
