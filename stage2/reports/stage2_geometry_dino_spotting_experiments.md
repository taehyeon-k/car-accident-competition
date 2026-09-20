# Stage 2 geometry-DINO spotting experiments

Run date: 2026-09-20. The screen used the fixed 201/50 train/validation split,
seed 0, frozen geometry-DINOv3 ViT-S features, and the checkpoint
`/workspace/outputs/geometry_pretrain/runs/phase1_partial_noanchor/backbone_best.pth`.
All numbers below are from saved result files. No leaderboard result is inferred.

## Outcome

The best method is **E3 ASFormer-lite**: two layers of local temporal attention,
192 hidden dimensions, a 17-position window, normalized soft event targets, and
random 96/128/160-position cadence augmentation. It scores **0.6344** on the
fixed validation split, versus **0.6215** for the strict FPS-blind E0 baseline
(+0.0129). Its four metrics are ENTRY 0.44, COLLISION 0.74, side Macro-F1
0.8599, and evasion Macro-F1 0.6162.

The strongest localization result is E1 classification+displacement. It reduces
ENTRY normalized MAE from 0.1100 to 0.0531 and COLLISION normalized MAE from
0.0328 to 0.0265 relative to E1 classification-only. Its competition-style
score is 0.6109 because its collision hit rate and evasion F1 are weaker.

## Baseline audit

The old cache was sampled at approximately 10 Hz with
`stride=round(native_fps/10)`. Training then constructed `frame_seconds`,
`entry_s`, and `collision_s`, and used Gaussian targets in seconds. The
private runtime instead selects 128 normalized positions directly from ordered
original frame numbers. Thus the old submission was FPS-blind only at runtime,
not during feature extraction, target construction, or loss.

The existing held-out selection result for that pipeline is 0.6100
(`submission_tools/fps_stage2/training_report.json`). Existing cadence stress
results fall from 0.6100 at cache stride 1 to 0.5300 at stride 4, quantifying a
material train/inference cadence sensitivity.

The new strict cache was extracted from 192 normalized positions for every
video. Its metadata records `fps_accessed: false`. Training resamples this base
grid to 96, 128, or 160 positions. Sampling, jitter, dropping, targets, loss,
checkpoint selection, and decoding use only frame order, original frame
numbers, and normalized clip position. FPS is read only by a separate evaluator
after predictions exist.

## Fixed-split results

| ID | Head / target | ENTRY | COLLISION | Side F1 | Evasion F1 | Score | Params |
|---|---|---:|---:|---:|---:|---:|---:|
| E3 | local attention / normalized soft | 0.44 | 0.74 | 0.860 | 0.616 | **0.634** | 1.04M |
| E0-fpsblind | dilated conv / normalized soft | **0.50** | 0.66 | 0.780 | **0.657** | 0.621 | 0.81M |
| E1 displacement | encoder-decoder / radius+offset | **0.52** | 0.60 | 0.859 | 0.600 | 0.611 | 1.18M |
| exact-anchor | baseline / exact CE | 0.42 | 0.68 | 0.880 | 0.618 | 0.610 | 0.81M |
| normalized 1% | baseline / normalized soft | 0.44 | 0.64 | 0.819 | 0.720 | 0.609 | 0.81M |
| E2 | TemporalMaxer / normalized soft | 0.34 | **0.76** | **0.880** | 0.594 | 0.606 | **0.56M** |
| E1 classification | encoder-decoder / normalized soft | 0.40 | 0.66 | 0.788 | 0.559 | 0.573 | 1.18M |
| E4 | event-conditioned TemporalMaxer | 0.38 | 0.74 | 0.653 | 0.548 | 0.572 | 0.76M |
| soft sampled-index | baseline / index soft | 0.40 | 0.60 | 0.779 | 0.678 | 0.569 | 0.81M |
| E2 displacement | TemporalMaxer / radius+offset | 0.26 | 0.68 | 0.778 | 0.720 | 0.554 | 0.56M |
| E3 displacement | local attention / radius+offset | 0.34 | 0.56 | 0.840 | 0.699 | 0.546 | 1.04M |

Random cadence matters: the comparable 15-epoch fixed-128 run scores 0.5840,
whereas the full randomized-cadence E0 scores 0.6215. The comparison is
directional because their horizons differ.

## E5 and robustness

For E3, 96/128/160-position inference scores 0.6165/0.6344/0.6211.
Mean normalized prediction standard deviation across views is 0.0173 for ENTRY
and 0.0059 for COLLISION. Median fusion scores 0.6104, so multi-view inference
does not justify its roughly 3x backbone cost. Use the single 128-position view.

Source diagnostics for E3 normalized MAE:

| Source | N | ENTRY | COLLISION |
|---|---:|---:|---:|
| AIHUB | 18 | 0.0350 | 0.0336 |
| CCD | 16 | 0.0855 | 0.0204 |
| NEXAR | 16 | 0.0685 | 0.0514 |

## Answers to the experiment questions

1. The main bottleneck is temporal event modeling, especially ENTRY. Local
   attention improves the overall score; TemporalMaxer is excellent for
   COLLISION but poor for ENTRY.
2. Classification+displacement clearly improves normalized localization inside
   E1, but does not improve the weighted score or transfer successfully to E2/E3.
3. TemporalMaxer-lite does not beat E3 or E0 overall. It is the best
   size/latency candidate and the best COLLISION localizer.
4. Strict FPS-blind training plus randomized cadence improves the direct
   fixed-split baseline and materially improves stability compared with the old
   cadence-sensitive cache. Fixed-cadence training is worse in the short sweep.
5. E3's gain over E0 comes mainly from COLLISION (+0.08) and side F1 (+0.08);
   ENTRY falls by 0.06.
6. Event-conditioned pooling hurts both categorical metrics in this setup.
7. Coarse-to-fine refinement was not run in the fast methodology sweep; E5
   already showed that extra backbone views do not pay off.
8. 14x20 spatial caching was not run because temporal and loss methods had not
   yet produced a large stable gain.
9. No geometry-tuned ViT-L checkpoint exists. The local ViT-L asset is V-JEPA,
   so no invalid S-vs-L comparison was made.
10. The requested CV matrix was stopped when the experiment objective was
    changed to a fast fixed-split methodology sweep. No statistical-stability
    claim is made from partial CV artifacts.

## Runtime and artifacts

Private-style raw-frame inference on the available RTX 5090 measured 3.49 s of
backbone time for three videos (384 encoded frames), 0.39 s total head time
including first-call overhead, and 729 MiB peak allocated VRAM. This is about
1.16 s/video for the backbone. A true L40S benchmark was unavailable in this
session; the older L40S-compatible runtime result suggests approximately
2.4 s per 128-frame video, so that number is an estimate rather than a new
measurement.

- Summary CSV: `stage2/spotting_experiments/results/fixed_split_summary.csv`
- Score plot: `stage2/spotting_experiments/results/fixed_split_scores.png`
- Best checkpoint: `stage2/spotting_experiments/results/E3-ASFormer/fixed/seed0/checkpoint.pt`
- Robustness/E5: `stage2/spotting_experiments/results/E3-ASFormer/fixed/seed0/robustness.json`
- Runtime benchmark: `stage2/spotting_experiments/results/E3-ASFormer/fixed/seed0/runtime.json`
- Cache documentation: `stage2/artifacts/spotting_features/fpsblind_192_7x10/meta.json`

Reproduce the winner:

```bash
cd /workspace/car-accident
python -m stage2.spotting_experiments.train \
  --config stage2/spotting_experiments/configs/e3_asformer.json \
  --output stage2/spotting_experiments/results/E3-ASFormer/fixed/seed0
```

Private-style inference:

```bash
python -m stage2.spotting_experiments.runtime DATA_DIR \
  --checkpoint stage2/spotting_experiments/results/E3-ASFormer/fixed/seed0/checkpoint.pt \
  --backbone-checkpoint /workspace/outputs/geometry_pretrain/runs/phase1_partial_noanchor/backbone_best.pth \
  --output predictions.csv --runtime-json runtime.json
```

The next experiment should target ENTRY specifically: combine E3's local
attention with separate ENTRY/COLLISION losses, keeping normalized soft
classification for COLLISION while applying displacement only to ENTRY. This is
better supported by the observed task split than spatial upscaling or more
test-time views.
