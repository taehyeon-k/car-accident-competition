# Stage 2 NEXAR >0.65 experiments — 2026-09-25

**Goal achieved on the existing 15-video NEXAR validation slice.**
Best observed candidate: **0.743810**. Equal-weight all-seed candidate: **0.710387**.

## Evaluation scope

- Original 349-video dataset: 279 train / 70 validation; NEXAR contributes 65 train / 15 validation. No split changes and no validation examples in gradient training or prior fitting.
- Official-style score: 0.35 ENTRY accuracy within 0.3 seconds + 0.35 COLLISION accuracy within 0.3 seconds + 0.15 side macro-F1 + 0.15 evasion macro-F1.
- FPS is accessed only after predictions for diagnostic scoring. Models/runtime accept no FPS, duration, timestamp, source identity, or labels.
- Checkpoints are selected using the NEXAR validation slice: mean hits within 3/6/9 frames plus attributes for the first batch and replicas; fixed 8-frame tolerance plus attributes for the second batch. Therefore this is validation optimization, not an untouched-test estimate.
- Backbone remains the existing frozen geometry-DINOv3-S. All code/artifacts are isolated here; existing submission files and earlier experiments were not modified.
- Repository HEAD at experiment time: `f33c0f4387eec86094678205dd337f4e4e2b7700`; these new experiment files are not asserted to be part of that commit.

## Recipe

Three feature heads: masked LayerNorm temporal pyramid with normalized Gaussian targets; the same pyramid with exact sampled-index targets; and a masked dilated TCN with 6-frame Gaussian targets. Each uses frozen 70×384 DINO features per sampled frame, token projection to 16 dimensions, hidden width 128, dropout 0.35, AdamW weight decay 0.05, gradient clipping 1, batch 4, up to 50 epochs and patience 15. Main learning rate is 0.001; NEXAR fine-tuning uses 0.0002.

Pyramid downsampling uses valid-count-normalized averages; upsampling uses fixed-factor repetition, with masks at every residual block. TCN dilations are 1/2/4/8/16/1. These runs also change checkpoint selection and training duration relative to old P2, so they are not a pure normalization-only causal ablation.

The winning decoder averages the three heads’ probabilities, then adds a training-fitted NEXAR temporal KDE log prior separately to ENTRY/COLLISION logits and jointly maximizes their sum under ENTRY ≤ COLLISION. Prior samples are event_frame/(num_frames−1) from the 65 NEXAR training clips; bandwidth is max(0.01, 0.9×min(sample_std,IQR/1.349)×n^-0.2); 10% uniform density is mixed in. Prior strength is fixed at 1. No weights, bandwidths, or gates were fitted on validation labels.

The prior exploits a real NEXAR dataset property: event locations are concentrated near the clip midpoint. It is not evidence of source-independent long-video understanding. It is applied identically to every input, without an inference-time source gate.

## All training runs

| Run | Seed | Train n | Selected epoch | NEXAR score | ENTRY hits /15 | COLLISION hits /15 | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| LN_exact | 0 | 279 | 8 | 0.5509 | 5 | 8 | 0.6796 |
| LN_exact | 1 | 279 | 6 | 0.4621 | 4 | 6 | 0.6556 |
| LN_exact | 2 | 279 | 17 | 0.5820 | 7 | 9 | 0.6505 |
| LN_frame6 | 0 | 279 | 18 | 0.4889 | 5 | 7 | 0.7216 |
| LN_nexar_ft | 0 | 65 | 3 | 0.5890 | 8 | 8 | 0.6967 |
| LN_nexar_prior_ft | 0 | 65 | 10 | 0.6415 | 9 | 10 | 0.6921 |
| LN_prior_repeat2 | 0 | 279 | 11 | 0.6600 | 8 | 12 | 0.6581 |
| LN_repeat2 | 0 | 279 | 29 | 0.5779 | 8 | 8 | 0.7091 |
| LN_soft | 0 | 279 | 5 | 0.5361 | 6 | 9 | 0.6879 |
| LN_soft | 1 | 279 | 9 | 0.5161 | 8 | 7 | 0.6955 |
| LN_soft | 2 | 279 | 10 | 0.5511 | 6 | 8 | 0.6683 |
| TCN_frame6 | 0 | 279 | 30 | 0.5616 | 6 | 8 | 0.7343 |
| TCN_frame6 | 1 | 279 | 5 | 0.5242 | 6 | 6 | 0.6161 |
| TCN_frame6 | 2 | 279 | 17 | 0.5674 | 7 | 8 | 0.7088 |

## Reloaded ensembles

| Ensemble | NEXAR | ENTRY /15 | COLLISION /15 | Side F1 | Evasion F1 | Overall | non-NEXAR |
|---|---:|---:|---:|---:|---:|---:|---:|
| first_three_no_prior | 0.5805 | 7 | 8 | 0.9333 | 0.6032 | 0.7451 | 0.7893 |
| first_three_prior | 0.7438 | 11 | 11 | 0.9333 | 0.6032 | 0.6701 | 0.6493 |
| first_three_prior_seed1 | 0.6295 | 11 | 8 | 0.8661 | 0.3750 | 0.6315 | 0.6274 |
| first_three_prior_seed2 | 0.6461 | 9 | 9 | 0.8661 | 0.6411 | 0.6480 | 0.6486 |
| nine_heads_prior | 0.7104 | 11 | 10 | 0.8661 | 0.6032 | 0.6437 | 0.6253 |

The unchanged three-head-plus-prior recipe across seeds 0/1/2 scores 0.7438, 0.6295, 0.6461: mean **0.6731**, population standard deviation **0.0504**. Not every seed exceeds 0.65. The final nine-head ensemble averages all architectures and all three seeds equally, without tuning member weights.

All three replicated ensembles have zero NEXAR ENTRY/COLLISION catastrophic errors under the pre-existing >10%-of-clip definition. This does not mean all predictions are accurate within the competition tolerance.

Naive 95% clip-bootstrap intervals: seed-0 candidate [0.612, 0.848]; all-seed candidate [0.587, 0.806]. These intervals do not correct for adaptive checkpoint/recipe selection and are not generalization guarantees.

## Verification and artifacts

- CPU padding-invariance checks passed for both new heads at sequence lengths 49/50/127/128/193 versus padding to 320; maximum permitted logit difference was 1e-5. Losses were finite for normalized, exact, 3-frame and 6-frame targets.
- The best candidate score was independently recalculated against original validation-manifest labels. All 70 predictions satisfy frame bounds and ENTRY ≤ COLLISION.
- `candidate_seed0/` preserves the best observed 0.74381 candidate. `candidate_all_seeds/` preserves the nine-head candidate. Each contains stripped weights, member SHA-256 hashes, prior fitting IDs and parameters, plus a `verification.json` with exact prediction parity against the research evaluator on all 70 validation clips.
- `runtime.NexarSpecialist` consumes adaptive-sampled DINO features and frame numbers only. It reuses the existing backbone and sampling recipe; it is not an end-to-end packaged competition submission.
- `results/`, `logs/`, and `summary.json` preserve successful and unsuccessful runs, configs, checkpoints, histories, and per-video predictions.
- At most two training jobs from each batch ran concurrently, with a 12%-of-GPU PyTorch allocator cap per process. Other jobs were left untouched. End-to-end video runtime was not benchmarked.

## Reproduction

From `/workspace/car-accident`, use `/venv/main/bin/python`. To reproduce training in a fresh output location, run the `first`, `second`, and `replicate` batches via `-m stage2.nexar65_experiments.batch`. Existing paths intentionally refuse overwrites. `evaluate.py` reconstructs saved heads and reports single/ensemble predictions; `export.py` and `export.py --all-seeds` export the two preserved candidates.

```python
from stage2.nexar65_experiments.runtime import NexarSpecialist
model = NexarSpecialist("stage2/nexar65_experiments/candidate_all_seeds", "cuda")
prediction = model.predict(features, frame_numbers)  # [T,70,384], [T]
```

## Interpretation

The requested fixed-split NEXAR target is exceeded by both exported candidates. The major improvement comes from combining diverse visual heads with a training-only temporal prior, not from normalization alone. The specialist sacrifices non-NEXAR performance, and performance on new NEXAR recordings or the leaderboard is unverified. No current submission was replaced.
