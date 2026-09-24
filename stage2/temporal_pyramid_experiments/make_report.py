"""Regenerate the requested evidence report from saved experiment artifacts."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("stage2/temporal_pyramid_experiments/results")
REPORT = Path("reports/stage2_temporal_pyramid_framecount_experiments.md")
RUNS = ["B0", "P1", "P2", "P3", "P4"]


def number(x, digits=3):
    return "—" if x is None else f"{x:.{digits}f}"


def row_metrics(x):
    if not x.get("n"): return f"{x.get('n', 0)} | — | — | — | — | — | — | —"
    return " | ".join([str(x["n"])] + [number(x.get(k)) for k in (
        "acc_entry_0.3s", "acc_collision_0.3s", "f1_entry_side_macro",
        "f1_evasion_space_macro", "competition_score", "entry_normalized_mae",
        "collision_normalized_mae")])


def main():
    data = json.loads((ROOT / "analysis.json").read_text())
    lines = ["# Stage 2 temporal-pyramid and frame-count experiments",
             "", "Frozen geometry-DINOv3-S, 349 videos (279 train / 70 validation), fixed manifest split, seed 0. All model inputs are FPS-, duration-, and timestamp-blind. The ±0.3-second metric below is computed only after frame-number predictions, using manifest FPS for offline evaluation; model selection uses a FPS-blind 1%-of-clip score. No leaderboard result is claimed.",
             "", "## Controlled design", "",
             "B0 is the existing 128-position ASFormer checkpoint (1,038,952 trainable parameters). P1–P4 share a 273,942-parameter 1D pyramid: 70×384 frozen DINO grid features → per-token 384→16 projection → 1120→128 frame projection → four stride-2 levels with residual depthwise-kernel-5/pointwise blocks → linear top-down fusion → dense ENTRY/COLLISION at the finest level and pooled side/evasion heads. Dropout 0.35. No FPS or duration is read by the dataset or head, and decoding maps normalized positions back to original frame numbers while enforcing ENTRY ≤ COLLISION.",
             "", "P1 always requests 128 positions. P2–P4 request all available frames for N≤128; 128 for 129–256; 192 for 257–512; 256 for 513–768; and 320 for N>768. P3 adds ±25% count variation, grid jitter, 2.5% frame drop, and a mild monotonic warp during training. P4 adds two-view CDF-L1 consistency at weight 0.05. All use the same frozen checkpoint and normalized Gaussian event targets (width 0.015), AdamW at 0.001, weight decay 0.05, one-cycle schedule, batch 4, up to 30 epochs, early-stop patience 7. The best FPS-blind validation checkpoint, not final weights, is evaluated.",
             "", "The 383-grid cache reused the old 192-grid features and encoded 17,329 additional unique frames. Its provenance is in `stage2/temporal_pyramid_experiments/cache_383/meta.json`; old cache/submission files were not changed.",
             "", "## Overall validation", "",
             "| Run | Score | ENTRY | COLLISION | Side F1 | Evasion F1 | Entry normalized MAE | Collision normalized MAE | Parameters |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for run in RUNS:
        x = data[run]["overall"]
        lines.append(f"| {run} | {number(x['competition_score'])} | {number(x['acc_entry_0.3s'])} | {number(x['acc_collision_0.3s'])} | {number(x['f1_entry_side_macro'])} | {number(x['f1_evasion_space_macro'])} | {number(x['entry_normalized_mae'])} | {number(x['collision_normalized_mae'])} | {data[run]['runtime']['parameter_count']:,} |")
    lines += ["", "## Frame-count bins", "",
              "Each cell is score (n); per-bin macro-F1 at n≤2 is highly unstable. There are no validation clips in 750–1000.", "",
              "| Run | <150 | 150–300 | 300–500 | 500–750 | 750–1000 | >1000 | Overall |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    labels = list(data["B0"]["bins"])
    for run in RUNS:
        x = data[run]
        cells = [f"{number(x['bins'][label].get('competition_score'))} ({x['bins'][label]['n']})" for label in labels]
        lines.append(f"| {run} | " + " | ".join(cells) + f" | {number(x['overall']['competition_score'])} |")
    lines += ["", "Detailed metrics for every nonempty bin:", "",
              "| Run | Bin | n | ENTRY | COLLISION | Side F1 | Evasion F1 | Score | Entry norm MAE | Collision norm MAE |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for run in RUNS:
        for label, x in data[run]["bins"].items():
            lines.append(f"| {run} | {label} | {row_metrics(x)} |")
    lines += ["", "## Sources and confounding", "",
              "| Run | AIHub (18) | CCD (17) | MM-AU (20) | Nexar (15) |",
              "|---|---:|---:|---:|---:|"]
    for run in RUNS:
        x = data[run]["sources"]
        lines.append(f"| {run} | " + " | ".join(number(x[s]["competition_score"]) for s in ("AIHUB", "CCD", "MMAU", "NEXAR")) + " |")
    lines += ["", "Nexar event hits (15 clips):", "",
              "| Run | ENTRY hits | COLLISION hits | Entry norm MAE | Collision norm MAE |",
              "|---|---:|---:|---:|---:|"]
    for run in RUNS:
        x = data[run]["sources"]["NEXAR"]
        lines.append(f"| {run} | {round(15*x['acc_entry_0.3s'])}/15 | {round(15*x['acc_collision_0.3s'])}/15 | {number(x['entry_normalized_mae'])} | {number(x['collision_normalized_mae'])} |")
    lines += ["", "There is almost no source/length overlap: validation AIHub clips are all 150 frames, CCD all 50, MM-AU spans 50–355 (median 102), and Nexar spans 540–1248 (median 1210). CausalCrash has one training clip and no validation clip. A within-source frame-count effect cannot be estimated reliably for three nearly fixed-length sources. Nexar has only 15 validation examples. Therefore the MM-AU/Nexar gap cannot be numerically decomposed into length versus domain shift on this split.",
              "", "Spearman correlation between frame count and a per-video correctness proxy (event hits plus attribute correctness, not macro-F1):", "",
              "| Run | Global rho | Nexar rho | Global entry-error rho | Global collision-error rho |",
              "|---|---:|---:|---:|---:|"]
    for run in RUNS:
        c = data[run]["correlations"]
        lines.append(f"| {run} | {number(c['global']['frame_count_vs_per_video_correctness_proxy_rho'])} | {number(c['NEXAR']['frame_count_vs_per_video_correctness_proxy_rho'])} | {number(c['global']['frame_count_vs_entry_normalized_error_rho'])} | {number(c['global']['frame_count_vs_collision_normalized_error_rho'])} |")
    lines += ["", "Full source-specific correlations are saved in `stage2/temporal_pyramid_experiments/results/analysis.json`. The plotted bin scores and validation curves are in [framecount_and_validation.png](../stage2/temporal_pyramid_experiments/results/framecount_and_validation.png). Dashed training curves are available for P1–P4; B0's historical run logged validation but not train scores each epoch.",
              "", "## Overfitting and resource cost", "",
              "| Run | Best epoch | Last epoch | Gap (epochs) | Train selection score at best weights | Val selection score | Head ms/video | Peak allocated VRAM MiB |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for run in RUNS:
        path = Path("stage2/spotting_experiments/results/E3-ASFormer-349/fixed/seed0") if run == "B0" else ROOT / run / "seed0"
        m = json.loads((path / "metrics.json").read_text())
        h = data[run]["history"]
        if run == "B0":
            best = max(h, key=lambda z: z["fpsblind_selection_score"])["epoch"]
            last = len(h); train = None
        else:
            best, last = m["best_epoch"], m["last_epoch"]
            train = m["train_fpsblind"]["fpsblind_selection_score"]
        rt = m["runtime"]
        lines.append(f"| {run} | {best} | {last} | {last-best} | {number(train)} | {number(m['fpsblind']['fpsblind_selection_score'])} | {number(rt['temporal_head_ms_per_video'])} | {number(rt['peak_vram_mb'],1)} |")
    lines += ["", "P1–P4 continue improving training metrics after validation peaks, confirming substantial overfitting despite the smaller head. In particular P1's best-weight train/validation selection scores are 0.870/0.444; P2's are 0.716/0.475. Early stopping is essential.",
              "", "## Sampling-density robustness", "",
              "For each video, predictions were compared at 96/128/192 positions (B0/P1) or 96/128/192/256/320 positions (P2–P4). Counts above the old 192 cache are not treated as independent encoded frames. Short videos are capped at native count for adaptive models.", "",
              "| Run | Score range | Entry normalized prediction SD | Collision normalized prediction SD | Entry mean frame range | Collision mean frame range |",
              "|---|---:|---:|---:|---:|---:|"]
    for run in RUNS:
        path = Path("stage2/spotting_experiments/results/E3-ASFormer-349/fixed/seed0") if run == "B0" else ROOT / run / "seed0"
        r = json.loads((path / "robustness.json").read_text())
        e, c = r["trajectories"]["entry"], r["trajectories"]["collision"]
        lines.append(f"| {run} | {number(r['score_range'])} | {number(e['mean_normalized_std'],4)} | {number(c['mean_normalized_std'],4)} | {number(e['mean_original_frame_range'],1)} | {number(c['mean_original_frame_range'],1)} |")
    lines += ["", "P3 lowers ENTRY view sensitivity versus P2, while P4 lowers COLLISION view sensitivity; neither produces a higher overall score. P4's score range is not lower than P3's, so the consistency loss has no clear net robustness benefit.",
              "", "A single long Nexar raw-frame benchmark using the same frozen DINO backbone and batch size 12 took 1.12 s for 128 unique frames and 2.00 s for 320, including frame loading/preprocessing and DINO feature extraction. The peak PyTorch-allocated VRAM was 405 MiB in both cases; CUDA context/reservation is additional. This is one warm-file benchmark under concurrent Stage 3 activity, not a competition runtime guarantee. Across validation, adaptive sampling encodes 143.7 unique frames/video versus 100.8 for fixed-128 deduplicated sampling (+42.6%); for >768-frame clips the cost is 320 versus 128 (2.5×).",
              "", "## Failure cases and explicit conclusions", "",
              "P2 misses long Nexar COLLISION by selecting unrelated peaks: `nexar_00200` predicts frame 113 versus GT 591, `nexar_00763` 1132 versus GT 591, and `nexar_00927` 1209 versus GT 608. Finer sampling did not prevent false-event selection. B0 also has large ENTRY misses on some of these clips. Details are in each run's `predictions.json`.",
              "", "- Is frame count independently causal? **Not identifiable** on this split; source and length have minimal overlap. The global score/length correlation is negative, but within Nexar it is unstable and sometimes reverses.",
              "- MM-AU versus Nexar after controlling for frame count? **Cannot estimate** without overlapping lengths or within-source length variation.",
              "- Pyramid at fixed 128? **No overall gain:** P1 0.674 versus B0 0.689; long COLLISION falls from 10/15 to 6/15.",
              "- Adaptive sampling on long ENTRY? **Yes, narrowly:** Nexar ENTRY 5/15→7/15 versus B0, but only 14 >1000-frame clips and no replication.",
              "- Adaptive sampling on long COLLISION? **No:** Nexar 10/15→5/15, outweighing the ENTRY gain for long clips.",
              "- Short-video harm? **No in P2:** <150 score 0.780→0.802; 150–300 score 0.686→0.790. Most of P2's overall gain comes from short/mid-length sources.",
              "- Temporal augmentation? **Some prediction stability, not score:** P3 overall 0.690; long score 0.488 versus P2 0.463.",
              "- Cross-view consistency? **Mixed:** P4 COLLISION prediction SD falls, but its overall score is 0.703 and density score range rises versus P3.",
              "- Flattest long-vs-short score curve? None solves the gap; B0 has the best >1000 score (0.492), with P3 close (0.488).",
              "- Replace current Stage 2 head? **No.** P2 has the best overall validation score (0.730) but worsens the key long-video/Nexar COLLISION failure and increases raw-frame cost. Single-seed differences on 70 validation videos do not establish leaderboard generalization.",
              "", "Best controlled experiment: **P2**, overall **0.730**, ENTRY **0.629**, COLLISION **0.786**, 273,942 trainable parameters, head **0.636 ms/video** plus frozen DINO, checkpoint `stage2/temporal_pyramid_experiments/results/P2/seed0/checkpoint.pt`. The temporal-pyramid-as-a-long-video-solution hypothesis is **not supported**, although P2 improved the aggregate score.", ""]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines))
    print(REPORT)


if __name__ == "__main__": main()
