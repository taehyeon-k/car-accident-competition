# Stage 2 geometry-DINOv3 experiments (prompt v2)

Run date 2026-09-20. Every number below comes from a saved file under
`stage2/experiments_v2/results/`. No leaderboard score is inferred or predicted.

Single run per experiment, fixed 201/50 split, **seed 42**, frozen geometry-tuned
DINOv3 ViT-S features from
`/workspace/outputs/geometry_pretrain/runs/phase1_partial_noanchor/backbone_best.pth`.
This is a methodology screen, not a submission run: there is no cross-validation
and no multi-seed averaging.

## Read this first: how large a difference is meaningful

The validation split has 50 clips, so one clip is 0.02 of ENTRY or COLLISION
accuracy and 0.007 of the weighted score. Re-running the submitted baseline under
two seeds gives **0.6174 (seed 0)** and **0.5852 (seed 42)** — a 0.032 swing from
the seed alone, with nothing else changed.

**Treat score gaps below roughly 0.03 as indistinguishable from noise.** Several
adjacent rows in the table below fall inside that band and should not be ranked
against each other. The conclusions drawn in this report rest only on gaps that
clear it, or on effects that reproduce across several sampling schemes.

## Existing baseline

`submission_tools/train_fps_stage2.py` — the pipeline behind the ~0.41 real
leaderboard score. Frozen geometry-DINOv3 ViT-S, letterboxed 448x800, patch grid
average-pooled to 7x10, cached at stride `round(native_fps/10)`. The head
(`submission_tools/fps_stage2/probe.py:TemporalProbe`, 0.81M parameters) projects
each token to 32 dimensions, flattens the 7x10 grid to a 192-dimensional frame
vector, applies two residual dilated 1D convolutions (k=5, dilations 1 and 2),
and reads out dense ENTRY/COLLISION logits plus attention-pooled side and evasion
logits. Training uses Gaussian targets in **seconds**
(`entry_sigma=0.15 s`, `collision_sigma=0.10 s`) and selects on a seconds-based
metric. Decoding is `constrained_decode`, which enforces ENTRY <= COLLISION.

FPS is not a network input, but it does enter the cache stride, the loss widths,
and checkpoint selection. That is the property prompt v2 set out to remove.

### Reproduction

Reproduced in `stage2/experiments_v2/baseline_submitted.py`, which imports the
original `TemporalProbe`, `normalized_indices`, `joint_loss` and
`constrained_decode` rather than copying them. It was **not** run through
`submission_tools/train_fps_stage2.py`, because that script writes into
`submission_tools/fps_stage2/` and would have overwritten the recorded submission
provenance. Only the held-out selection phase is reproduced; the submitted script
then refits on all 251 clips, which trains on the validation clips and therefore
has no independent score to compare.

| Baseline | ENTRY | COLL | side F1 | evasion F1 | Score |
|---|---:|---:|---:|---:|---:|
| Recorded `training_report.json` | 0.40 | 0.70 | 0.8199 | 0.6800 | 0.6100 |
| Reproduction, seed 0 | 0.46 | 0.70 | 0.8599 | 0.5495 | 0.6174 |
| Reproduction, seed 42 | 0.40 | 0.64 | 0.8182 | 0.6566 | 0.5852 |

The seed-0 reproduction lands within 0.008 of the recorded number, so the
regenerated feature cache and the reproduced protocol are sound.

One caveat: the probe feature cache
(`/workspace/cache/geometry_pretrain/stage2_probe_features/adapted_noanchor/`) was
absent from both this machine and R2 — the bucket's own `upload_r2.sh` excludes
it — so all 251 files were re-extracted from frames for this report.

## Changes implemented

New package `stage2/experiments_v2/`. Nothing under `stage2/spotting_experiments/`
or anywhere else was modified; the FPS-blind feature cache is read read-only.

| Component | Prompt | File |
|---|---|---|
| Multi-rate temporal differences `D_s[t]=F[t]-F[t-s]`, gated fusion, boundary-clamped | §6 | `modules.py:MultiRateDifference` |
| Event-state auxiliary heads, BEFORE/BOUNDARY/AFTER and PRE/CONTACT/POST, radius in sampled positions | §9 | `objective.py:state_aux_loss` |
| Two-view temporal consistency, Jensen-Shannon over probability mass binned into shared original-frame coordinates | §12 | `objective.py:consistency_loss` |
| Local temporal warping, frame duplication, random drop, random cadence | §11 | `sampling.py` |
| P25/P50/P75/P90 error distributions | §17 | `metrics.py:error_distribution` |
| Sampling-density instability report | §18 | `robustness.py` |
| Directional and conditional failure breakdown | §21 | `failure_analysis.py` |

Existing designs reused rather than reinvented: dense per-frame event heads, the
three temporal-head candidates, soft normalized targets, and event-conditioned
readout. The whole matrix was re-run under one harness so every row is comparable.

### FPS-blindness

No FPS, duration, timestamp or seconds value reaches sampling, target
construction, the model, the loss, checkpoint selection, or decoding. Original
frame numbers are carried only so predictions come back as original frame
indices. `native_fps` is read in exactly one place, `metrics.py:official_metrics`,
after predictions exist. `tests/test_v2.py::test_training_path_reads_no_fps_field`
enforces this by scanning executable source with comments and string literals
stripped, so prose cannot satisfy it. 23 CPU tests pass.

## Experiment table

Fixed 128-position evaluation view, the same 50 validation clips throughout.

| Experiment | ENTRY | COLL | side F1 | evasion F1 | Score | Params |
|---|---:|---:|---:|---:|---:|---:|
| E2c local attention | 0.48 | 0.82 | 0.779 | 0.643 | **0.6683** | 1.04M |
| E8 event-conditioned | 0.42 | 0.80 | 0.820 | 0.619 | 0.6428 | 1.62M |
| E7 temporal consistency | 0.42 | 0.74 | 0.860 | 0.694 | 0.6391 | 1.41M |
| E4 soft targets | 0.38 | 0.76 | 0.880 | 0.635 | 0.6263 | 1.41M |
| E6 sampling augmentation | 0.42 | 0.74 | 0.839 | 0.576 | 0.6182 | 1.41M |
| *Baseline, seed 0* | *0.46* | *0.70* | *0.860* | *0.550* | *0.6174* | *0.81M* |
| E2b TemporalMaxer | 0.40 | 0.66 | 0.860 | 0.594 | 0.5891 | 0.67M |
| E5 state aux | 0.40 | 0.70 | 0.778 | 0.579 | 0.5884 | 1.41M |
| E3 multi-rate diff | 0.38 | 0.64 | 0.900 | 0.612 | 0.5839 | 1.41M |
| *Baseline, seed 42* | *0.40* | *0.64* | *0.818* | *0.657* | *0.5852* | *0.81M* |
| E1 dense event head | 0.28 | 0.62 | 0.860 | 0.740 | 0.5550 | 0.44M |
| E2a dilated conv | 0.24 | 0.62 | 0.880 | 0.740 | 0.5440 | 0.81M |
| B0 pooled baseline | 0.22 | 0.14 | 0.797 | 0.700 | 0.3505 | 0.48M |

Localization error, best three configurations:

| Run | ENTRY p50 | ENTRY p90 | COLL p50 | COLL p90 |
|---|---:|---:|---:|---:|
| E2c | 5.0 fr / 0.0372 | 59.3 fr / 0.2339 | 1.0 fr / 0.0067 | 8.0 fr / 0.0403 |
| E7 | 6.0 fr / 0.0336 | 40.2 fr / 0.2265 | 2.0 fr / 0.0134 | 19.7 fr / 0.1083 |
| E8 | 7.0 fr / 0.0408 | 77.9 fr / 0.2449 | 2.0 fr / 0.0124 | 18.0 fr / 0.0757 |

The mean ENTRY error (25-38 frames) is an order of magnitude above the median
(5-7 frames). A minority of clips miss completely and drag the mean; most clips
are close. ENTRY p90 near 0.23 normalized means the worst tenth is essentially a
failure, not a near miss.

## FPS-invariance robustness

Each trained model re-evaluated under six sampling schemes, predictions mapped
back to original-frame coordinates. This is the decision-relevant table, because
the private videos have unknown and varying FPS.

| Run | std128 | dense192 | 96 | drop20 | drop40 | warp | **mean** | **worst** | range |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E7 consistency | 0.6391 | 0.6255 | — | 0.6054 | 0.5365 | 0.5935 | **0.6088** | **0.5365** | 0.1166 |
| E8 event-cond | 0.6428 | 0.5770 | — | 0.6008 | 0.5429 | 0.5419 | 0.5906 | 0.5419 | **0.1009** |
| E2c attention | **0.6683** | 0.6163 | 0.6099 | 0.5859 | 0.4861 | 0.5315 | 0.5830 | 0.4861 | 0.1822 |
| E6 augmentation | 0.6182 | 0.5972 | — | 0.5617 | 0.5314 | 0.5002 | 0.5648 | 0.5002 | 0.1180 |
| E3 multi-rate | 0.5839 | 0.5574 | — | 0.5499 | 0.4890 | 0.5254 | 0.5471 | 0.4890 | 0.0949 |

**The ranking inverts between the fixed view and the robustness average.** E2c
wins the single 128-position view by 0.026 over E8 and 0.029 over E7 — both
inside the noise band — but is the most fragile model tested, losing 0.182
between its best and worst scheme and collapsing to 0.4861 under 40% frame drop.
E2c trains at a single fixed cadence, so this is unsurprising in hindsight.

E7 leads on mean-across-schemes and its worst case is 0.050 above E2c's, which is
larger than the seed-noise band. Prediction instability for E7 is 0.0470 mean
normalized standard deviation for ENTRY and lower for COLLISION.

## Ablation analysis

**Dense per-frame event prediction is the one large, unambiguous win.**
B0 -> E1 is +0.205, almost entirely COLLISION 0.14 -> 0.62. Regressing a single
pooled scalar per event is simply the wrong output parameterization. This
confirms prompt §8 decisively.

**Local attention is the best temporal head.** E2a dilated 0.5440, E2b
TemporalMaxer 0.5891, E2c local attention 0.6683. E2c over E2a is +0.124, well
clear of noise. E2b vs E2a is not.

**Multi-rate temporal differences did not work here.** E2c -> E3 is **-0.084** at
the fixed view and -0.036 on the robustness mean. Prompt §6 called this "one of
the most important experiments"; on this data, with this backbone, adding
strides [1,2,4,8] on top of local attention hurt. A plausible reading is
redundancy: local attention over a 33-position window can already form temporal
contrasts, so the explicit difference channels mostly add parameters (+0.37M) and
overfitting surface on 201 training clips. This was not isolated further.

**The chain after E3 recovers, but never fully.** Because the prompt's matrix
builds E4-E8 on E3, every later row inherits the E3 regression. Soft targets
(+0.042), sampling augmentation (+0.034) and consistency (+0.021 over E6) each
help, and E8 reaches 0.6428 — still below plain E2c. **Whether these components
help on top of E2c directly is untested and is the single most valuable
follow-up.**

**Event-state auxiliary supervision did not help.** E3 -> E5 is +0.005, noise.

**Sampling augmentation trades peak for stability.** E3 -> E6 is +0.034 at the
fixed view, and it cuts nothing from the range; the consistency loss on top (E7)
is what produces the best worst-case behaviour.

**Event-conditioned heads are ambiguous.** E7 -> E8 is +0.004 at the fixed view
and -0.018 on the robustness mean; side F1 falls 0.860 -> 0.820 and evasion F1
0.694 -> 0.619. No evidence of benefit.

## Failure analysis

E7, 50 validation clips, tolerance set per clip to the offline ±0.3 s equivalent.

- ENTRY: 21 hit, 14 early, 15 late. Errors are roughly symmetric, so this is
  imprecision, not a systematic bias.
- COLLISION: 37 hit, 8 early, 5 late.
- entry_side wrong on 7 clips, evasion_space wrong on 15. Evasion is the weakest
  categorical target in every configuration.

Conditional breakdown, ENTRY normalized error:

| Condition | group | rest |
|---|---:|---:|
| tight ENTRY→COLLISION gap (<=10 frames, n=11) | 0.1105 | 0.0752 |
| short clip (<200 frames, n=34) | 0.0904 | 0.0671 |
| early event (position < 0.4, n=6) | 0.0874 | 0.0823 |

Per source, ENTRY / COLLISION normalized MAE: AIHUB 0.0574 / 0.0522, CCD
0.1276 / 0.0242, NEXAR 0.0671 / 0.0260.

Reading: ENTRY failures concentrate where ENTRY and COLLISION are close together
and where clips are short — that is, where the two events must be separated at
fine temporal resolution. CCD is the hardest source for ENTRY by a factor of two
while being among the easiest for COLLISION, which points at event semantics
(when exactly a vehicle "enters") rather than image quality. Event position
inside the clip has no real effect, which is the desired outcome: the model is
not exploiting where accidents usually sit.

COLLISION is largely solved at median (1-2 frames); ENTRY is not. The weighted
score is now limited by ENTRY and by evasion_space.

## Runtime

RTX 5090, cached features, head only.

| Run | wall | head | peak VRAM | params |
|---|---:|---:|---:|---:|
| E2c | 119 s | 0.20 ms/video | 567 MiB | 1.04M |
| E7 | 161 s | 0.21 ms/video | 929 MiB | 1.41M |
| E8 | 166 s | 0.26 ms/video | 932 MiB | 1.62M |

Backbone cost is excluded; the earlier measurement of about 1.16 s/video for
DINOv3-S on this machine still dominates inference.

## Recommended final architecture

**Continue with E7: local attention + sampling augmentation + two-view temporal
consistency**, and drop the multi-rate difference block.

The reasoning is the robustness table, not the headline score. E2c's 0.6683 is
the best single number but it comes from the one evaluation condition the prompt
explicitly warns is unrepresentative, and E2c degrades furthest when sampling
density changes. Under unknown private FPS, E7's mean 0.6088 and worst case
0.5365 are the better bet, and its advantage in the worst case exceeds the
seed-noise band while E2c's advantage at the fixed view does not.

This supports the prompt's central hypothesis in a qualified form: geometry-aware
per-frame DINOv3-S features plus a lightweight, FPS-blind event-transition head
are competitive, and on this validation split they beat the reproduced submitted
baseline at matched seed 42 (E7 0.6391 vs 0.5852). But the specific mechanism the
prompt expected to matter most, multi-rate differences, is what failed.

Immediate next steps, in order of expected value:

1. Re-run E4, E6, E7 with the multi-rate difference block removed, i.e. layered
   directly on E2c. The matrix confounded "diffs" with everything after it.
2. Robustness-test the reproduced baseline on the same six schemes. It uses a
   different cache pipeline, so it is currently the only row without a
   robustness number, and the comparison is incomplete without it.
3. Target ENTRY specifically on tight-gap clips: a separate ENTRY loss, or the
   high-resolution second pass from §14, which was not run.
4. Before acting on anything in the 0.03 band, re-run the top three
   configurations over several seeds. The screen cannot separate them.

Not run: §14 high-resolution refinement (needs a second feature cache at a finer
grid) and §15 backbone unfreezing (the cache is frozen by construction). Both are
recorded as untested rather than as negative results.

## Artifacts

- Summary: `stage2/experiments_v2/results/summary.csv`
- Per run: `config.json` (git commit, seed, exact train/val ids), `history.json`,
  `predictions.json`, `metrics.json`, `checkpoint.pt`
- Robustness: `results/<run>/robustness.json`
- Failure analysis: `results/<run>/failure_analysis.json`
- Baseline: `results/BASELINE_submitted_seed{0,42}/report.json`

```bash
cd /workspace/car-accident
python -m stage2.experiments_v2.make_configs
python -m stage2.experiments_v2.train --config stage2/experiments_v2/configs/E7_consistency.json \
  --output stage2/experiments_v2/results/E7_consistency
python -m stage2.experiments_v2.summarize
python -m stage2.experiments_v2.robustness \
  --checkpoint stage2/experiments_v2/results/E7_consistency/checkpoint.pt \
  --output stage2/experiments_v2/results/E7_consistency/robustness.json
python -m stage2.experiments_v2.failure_analysis --run stage2/experiments_v2/results/E7_consistency
python -m stage2.experiments_v2.baseline_submitted --seed 42 \
  --output stage2/experiments_v2/results/BASELINE_submitted_seed42
PYTHONDONTWRITEBYTECODE=1 python -m pytest stage2/experiments_v2/tests -q
```
