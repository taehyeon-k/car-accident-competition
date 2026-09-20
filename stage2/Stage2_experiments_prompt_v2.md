You are working on my DACON car accident competition Stage 2 codebase.

The goal is to improve my current Stage 2 architecture through controlled experiments.

## Current leaderboard state

My real leaderboard results are approximately:

* Complex joint model using geometry model + V-JEPA v2.1 ViT-L + DINOv3 ViT-B: **0.44**
* Geometry-tuned DINOv3 ViT-S + simple prediction head: **0.41**

The simple DINOv3-S model is therefore surprisingly close to the much more complicated model.

I want to build on the geometry-tuned DINOv3 ViT-S rather than return to a large video backbone.

---

# 1. Critical Stage 2 constraints

Stage 2 predicts four targets:

1. `entry_frame`
2. `collision_frame`
3. `entry_side`
4. `evasion_space`

The temporal targets dominate the score:

* ENTRY frame: 0.35
* COLLISION frame: 0.35
* entry side: 0.15
* evasion space: 0.15

Therefore precise event localization is the main objective.

## FPS constraint

The private videos have unknown and varying FPS.

FPS and video duration must NOT be used as model inputs.

The model must not assume that:

* 5 frames = some fixed amount of time
* 32 frames = some fixed duration
* ENTRY→COLLISION frame distance has a fixed physical meaning

DACON only uses hidden FPS/time metadata during evaluation to determine whether the predicted original frame is within ±0.3 seconds of GT.

Therefore:

**Design the architecture to operate entirely in frame/order/visual-state space.**

The model should learn visual state transitions such as:

ENTRY:

```text
victim outside ego lane
→ wheel approaches lane boundary
→ first lane contact
→ victim inside ego lane
```

COLLISION:

```text
separate vehicles
→ rapidly closing
→ first contact
→ post-impact state
```

Do not explicitly convert frames to seconds anywhere in the model.

---

# 2. Core architectural hypothesis

Use:

```text
geometry-tuned DINOv3 ViT-S
+
lightweight high-resolution temporal event spotter
```

Do NOT initially use:

* V-JEPA
* 3D CNNs
* large video transformers
* ViT-L
* complex detector/tracker graph architectures

The image backbone should extract strong per-frame geometry.

Temporal modeling should primarily answer:

> Between which neighboring visual states did ENTRY or COLLISION occur?

---

# 3. First inspect the existing code

Before changing anything:

1. Find the existing Stage 2 geometry-tuned DINOv3 ViT-S implementation.
2. Understand:

   * preprocessing
   * temporal sampling
   * DINO feature extraction
   * current simple head
   * losses
   * target representation
   * train/val split
   * evaluation
   * inference/submission generation
3. Preserve the existing working baseline.
4. Make the new architecture selectable through configuration so baseline comparisons remain possible.

Do not rewrite unrelated parts of the codebase.

---

# 4. Proposed architecture

Implement an architecture approximately like:

```text
Frames
  ↓
Geometry-tuned DINOv3 ViT-S
  ↓
Per-frame spatial representation
  ↓
Multi-rate temporal difference features
  ↓
Lightweight temporal encoder / event spotter
  ↓
ENTRY probability per frame
COLLISION probability per frame
  ↓
Event-conditioned classification
  ↓
entry_side
evasion_space
```

---

# 5. Preserve dense DINO information

Do not immediately compress every frame to only one CLS token.

Extract both:

```text
global/CLS feature
+
compressed spatial patch feature
```

Possible approach:

* take DINO patch tokens
* spatially pool them into a small grid such as:

  * 3×5
  * 4×7
  * similarly inexpensive spatial representation
* project them to a reasonable dimension

The purpose is to preserve:

* lane boundaries
* victim vehicle geometry
* wheel/lane relationship
* surrounding road space
* nearby vehicles/barriers

Keep memory consumption reasonable.

---

# 6. Multi-rate temporal difference features

This is one of the most important experiments.

For per-frame feature `F[t]`, construct differences such as:

```text
D1[t] = F[t] - F[t-1]
D2[t] = F[t] - F[t-2]
D4[t] = F[t] - F[t-4]
D8[t] = F[t] - F[t-8]
```

Use only available previous frames near sequence boundaries.

Fuse:

```text
F[t]
D1[t]
D2[t]
D4[t]
D8[t]
```

through projection/gating/attention.

Do not interpret these strides as seconds.

They represent multiple frame-space temporal scales.

Make the stride set configurable.

Default:

```text
[1, 2, 4, 8]
```

---

# 7. Lightweight temporal head

Build a lightweight temporal model.

Preferred candidates:

### Candidate A

T-DEED/SGP-inspired temporal blocks.

### Candidate B

TemporalMaxer-style lightweight temporal aggregation.

### Candidate C

A small 1D temporal transformer using only relative/local temporal reasoning.

Do not build a large transformer.

Target scale:

* approximately 2–4 temporal blocks initially
* modest hidden dimension
* local/multi-scale reasoning
* no strong absolute-frame-position dependence

Avoid learned absolute temporal embeddings that may cause the model to memorize where accidents normally happen in a clip.

Relative position information is acceptable.

---

# 8. Predict dense event distributions

Do NOT directly regress one normalized scalar frame position.

Predict:

```text
P_entry[t]
P_collision[t]
```

for every candidate frame.

Final frame prediction:

```text
argmax(P_entry)
argmax(P_collision)
```

must map back exactly to the corresponding original video frame index.

Never lose the original-frame mapping during sampling/cropping.

---

# 9. Event-state auxiliary supervision

Add optional auxiliary state classification.

For ENTRY:

```text
BEFORE_ENTRY
ENTRY_BOUNDARY
AFTER_ENTRY
```

For COLLISION:

```text
PRE_CONTACT
CONTACT
POST_CONTACT
```

Generate these auxiliary labels from the GT event frame using configurable frame-space neighborhoods.

The purpose is to force the model to understand the state transition instead of memorizing event location.

Make these losses individually switchable.

---

# 10. Soft event supervision

Experiment with soft event targets instead of only one-hot supervision.

For a GT event frame `g`, nearby frames can receive decreasing probability based on frame distance.

Possible distributions:

* Gaussian
* triangular
* exponential

Example concept:

```text
target[t] = exp(-abs(t-g) / tau)
```

Important:

`tau` must be defined in frames, not seconds.

Compare:

1. one-hot CE/BCE
2. soft target CE/KL/BCE

Keep both implementations configurable.

---

# 11. FPS-invariance augmentation

Implement explicit temporal-sampling invariance training.

This is a high-priority experiment.

From one training video, produce temporally modified views through:

### Random frame dropping

Example:

```text
original:
0 1 2 3 4 5 6 7 8 9

view:
0 1 3 4 6 7 9
```

### Random stride sampling

Use different frame-space sampling strides.

### Local temporal warping

Different portions of the clip may be sampled at different densities.

### Optional frame duplication

Randomly duplicate selected frames.

Important:

Each sampled frame must retain its original frame index.

GT ENTRY and COLLISION labels must be correctly remapped.

---

# 12. Temporal consistency loss

For two augmented temporal views of the same video:

```text
view A
view B
```

run both through the model.

Map their event probability distributions back into original-frame coordinates.

Where both views contain corresponding original frames, encourage:

```text
P_entry_A ≈ P_entry_B
P_collision_A ≈ P_collision_B
```

Possible losses:

* KL divergence
* Jensen-Shannon divergence
* MSE on logits/probabilities

The objective is:

> prediction should remain stable even when the temporal sampling density changes.

Make consistency weight configurable.

---

# 13. Event-conditioned classification heads

Do not classify `entry_side` and `evasion_space` only from global video pooling.

Use event-aware representations.

For ENTRY:

```text
z_entry = weighted temporal feature around P_entry
```

Use it for:

```text
entry_side
```

For COLLISION / surrounding scene:

```text
z_collision = weighted temporal/spatial feature around P_collision
```

Use it for:

```text
evasion_space
```

A simple implementation can use probability-weighted feature aggregation:

```text
z_entry = Σ P_entry[t] * Z[t]
z_collision = Σ P_collision[t] * Z[t]
```

Optionally include a small local frame-space neighborhood around the predicted event.

Again: no seconds.

---

# 14. Optional high-resolution refinement

Implement this only after the base temporal experiments work.

Use a two-pass system:

### Pass 1

Run all sampled frames through DINOv3-S at moderate resolution.

Produce ENTRY/COLLISION probabilities.

### Pass 2

Select top-K candidate temporal regions.

Re-run only those candidate frames at higher spatial resolution.

Use refined spatial features to improve exact event localization.

This should be especially useful for ENTRY because the discriminating evidence may be a small wheel/lane-boundary interaction.

Make this optional due to additional computation.

---

# 15. Training strategy

Because the labeled dataset is relatively small, prioritize controlled fine-tuning.

Test:

### Experiment 1

Frozen geometry-tuned DINOv3-S + train temporal/event heads only.

### Experiment 2

Unfreeze only final DINO blocks.

### Experiment 3

Small learning rate full fine-tuning only if previous experiments clearly justify it.

Use lower LR for the backbone than the new heads.

Prevent unnecessary backbone overfitting.

---

# 16. Ablation experiments

Run experiments incrementally.

The minimum experiment matrix should include:

### Baseline

Current geometry-tuned DINOv3-S + simple head.

### E1

DINOv3-S + dense per-frame event head.

### E2

E1 + lightweight temporal head.

### E3

E2 + multi-rate differences `[1,2,4,8]`.

### E4

E3 + soft event labels.

### E5

E3 + event-state auxiliary loss.

### E6

E3 + random FrameDrop / stride augmentation.

### E7

E6 + temporal consistency loss.

### E8

E7 + event-conditioned SIDE / EVASION heads.

### E9

Best model + optional high-resolution candidate refinement.

Do not change many components simultaneously before establishing their individual effect.

---

# 17. Evaluation diagnostics

In addition to the competition metric, report separately:

```text
ENTRY accuracy under competition tolerance
COLLISION accuracy under competition tolerance
entry_side F1
evasion_space F1
overall Stage 2 score
```

Also report frame localization error distributions.

For each temporal event report:

```text
median absolute frame error
mean absolute frame error
P25 / P50 / P75 / P90 error
```

Since training data may have FPS metadata but inference architecture must be FPS-independent, do not optimize the model using FPS.

If validation FPS happens to exist, it may only be used in a clearly separated diagnostic evaluator that reproduces DACON's ±0.3 second metric.

It must never enter the network or preprocessing decisions used at inference.

---

# 18. Important robustness analysis

Test the same validation videos under different temporal sampling densities.

Examples:

```text
all frames
every 2nd frame
random 20% frame drop
random 40% frame drop
piecewise irregular sampling
```

After mapping predictions back to original-frame coordinates, measure how much the predicted event changes.

Report something like:

```text
ENTRY prediction instability
COLLISION prediction instability
```

The best model should remain stable under sampling-density changes.

This robustness test is important because the private data has unknown FPS.

---

# 19. Avoid leakage

Audit the model for accidental temporal leakage.

The model must NOT use:

```text
FPS
duration seconds
timestamps
PTS expressed as physical time
fixed seconds-per-window
seconds-to-frame conversion
dataset-specific FPS assumptions
```

Also avoid excessive reliance on normalized absolute temporal position.

Original frame indices may be tracked internally only to restore output predictions to the required submission frame number.

---

# 20. Engineering requirements

Please:

1. Keep current baseline runnable.
2. Add clean configs for each experiment.
3. Reuse cached DINO features when possible.
4. Add deterministic seeds.
5. Log experiments to the existing experiment logging system/W&B if available.
6. Record:

   * config
   * git commit
   * train loss
   * validation metrics
   * model parameter count
   * runtime
   * GPU memory
7. Save best checkpoints by validation Stage 2 metric.
8. Ensure inference outputs valid original frame indices.

---

# 21. Final deliverable

After implementing and running the experiments, produce a Markdown report:

```text
stage2_dinov3_experiments_report.md
```

The report should contain:

## Existing baseline

Explain exactly how the current geometry-tuned DINOv3-S model works.

## Changes implemented

Describe every new module.

## Experiment table

Example:

| Experiment | Entry | Collision | Side | Evasion | Overall |
| ---------- | ----: | --------: | ---: | ------: | ------: |

## FPS-invariance robustness

Compare prediction stability under different temporal subsampling schemes.

## Ablation analysis

Explain which modifications actually improve performance.

## Failure analysis

Inspect examples where:

* ENTRY is early/late
* COLLISION is early/late
* SIDE is wrong
* EVASION is wrong

Determine whether failures appear related to:

* small-object spatial detail
* insufficient temporal discriminability
* occlusion
* unusual road geometry
* temporal sampling density
* wrong event semantics

## Recommended final architecture

Based on evidence from the experiments, identify the strongest architecture to continue developing.

Do not assume the most complicated architecture is best.

The primary hypothesis to test is:

> Strong geometry-aware per-frame DINOv3 features combined with lightweight, multi-scale, FPS-blind event-transition modeling can outperform a much more complicated video architecture for this Stage 2 task.
