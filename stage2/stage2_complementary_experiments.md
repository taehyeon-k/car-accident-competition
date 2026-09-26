

Objective

The goal is to determine whether additional physically meaningful or temporally structured signals can provide complementary improvements beyond the current v5 model.

Current strong components already include:

frozen geometry-tuned DINOv3-S

LCPyramid temporal head

phase-supervised members

learned motion-input members

global camera-shift collision fusion

native-frame collision snapping

probability ensemble.

Do NOT perform a general architecture search.

Do NOT introduce:

a larger video backbone

Transformer/SSM replacements

a new detector-heavy inference pipeline

expensive optical flow at inference

FPS/duration/timestamp dependence.

The experiments should isolate the effectiveness of four candidate signals:

collision-risk progression supervision

victim/ego-lane intrusion supervision

ego-motion-compensated residual motion

explicit event-boundary/change-point supervision.

The main question is:

Does each signal add information that is complementary to the existing geometry + motion + phase system?

Common experimental rules

Keep fixed whenever possible:

DINOv3-S geometry backbone

frozen visual features

adaptive sampling

LCPyramid architecture

optimizer/scheduler

train/val folds

current direct ENTRY/COLLISION losses

side/evasion heads

existing motion fusion

inference FPS-blindness.

Create a clean module such as:

stage2/aux_signal_experiments/

Do not overwrite existing checkpoints/results.

Use the existing 349-video dataset and existing 5-fold CV splits.

For each auxiliary signal, always compare against an exactly matched control.

Primary evaluation should be the incremental gain when inserted into the current v5-style ensemble, not only standalone model score.

Baselines

Reproduce or reuse:

B0

Current no-extra-signal LC-v2 control.

B1

Current PH_phase model.

B2

Current M_motion model.

B3

Current ensemble:

C0 + X_ema + M_motion + PH

with the existing:

camera-shift collision fusion

native-frame snap.

This is the practical reference.

Experiment 1 — Collision-risk progression supervision

Hypothesis:

The hidden representation should model not only discrete accident phase, but also continuous progression of collision risk.

Add a lightweight per-position risk head:

r_t = sigmoid(MLP(h_t)).

Do not remove the direct ENTRY/COLLISION heads.

Test risk supervision only as an auxiliary task first.

Construct a target that increases toward COLLISION using only frame order and GT event positions.

Do NOT use FPS or seconds.

Test at least:

R1 — simple progression target

Low risk before ENTRY, gradually increasing between ENTRY and COLLISION, high near COLLISION.

For example, normalized sequence/event-relative progression.

R2 — future-consistency supervision

Encourage later predictions to contain at least as much collision-risk evidence as nearby earlier predictions.

Use stop-gradient on the future target when appropriate.

R3 — adaptive monotonic regularization

Penalize obvious large decreases in predicted collision risk before COLLISION, but do not require strict monotonicity at every position.

Keep this lightweight.

Evaluate:

standalone score

ENTRY hit

COLLISION hit

collision normalized MAE

catastrophic collision errors

whether COLLISION candidate Recall@1 improves.

Also test whether adding the risk-supervised model to the v5 ensemble improves the ensemble.

Experiment 2 — Victim / ego-lane intrusion supervision

Hypothesis:

The current system has a strong collision-specific physical cue, but lacks an equally explicit ENTRY-specific cue.

The auxiliary target should describe the victim vehicle's relation to the ego lane.

Desired conceptual states:

OUTSIDE -> APPROACHING -> CROSSING -> INSIDE

or a simpler binary/ordinal equivalent if pseudo-label quality is insufficient.

The most important transition is:

APPROACHING/OUTSIDE -> CROSSING

around ENTRY.

Target generation

First inspect what geometry models / pseudo-label infrastructure already exists in the repository.

Reuse existing lane, vehicle, geometry or segmentation tools if possible.

Do not add a heavy model into final inference.

Pseudo-labeling can be done offline.

Possible pseudo-label information:

ego lane boundaries

victim vehicle box/mask

victim wheel or bottom-center location

overlap/distance between victim and ego lane

lateral distance to lane boundary

whether the vehicle has crossed into ego lane.

Before training:

inspect pseudo-label quality visually on a representative sample;

quantify missing/invalid labels;

reject this experiment if pseudo-labels are clearly unreliable.

Test:

L1 — categorical lane-state auxiliary head

Predict:
outside / approaching / crossing / inside.

L2 — ordinal lane-intrusion score

Predict a scalar or ordinal measure of how far the victim has progressed toward/into the ego lane.

L3 — ENTRY transition supervision

Add a small auxiliary loss focused on the transition into the crossing state.

Use these only as training supervision initially.

Do not use pseudo-label geometry directly in the final decoder unless the auxiliary results are clearly positive.

Main metrics:

ENTRY hit rate

ENTRY normalized MAE

ENTRY candidate Recall@K

long-clip ENTRY performance

overall Stage-2 score.

This experiment should be considered successful only if ENTRY improves without a major COLLISION regression.

Experiment 3 — Ego-motion-compensated residual motion

Hypothesis:

Current global camera shift is a useful COLLISION cue, but it mixes:

normal ego motion

road vibration

turning

impact

independent vehicle motion.

Create an additional cheap motion representation that measures motion remaining after dominant camera motion is removed.

Reuse the current native-frame motion preprocessing as much as possible.

Do not introduce RAFT/SEA-RAFT into the Stage-2 runtime.

Test a lightweight pipeline such as:

estimate dominant global translation or affine motion;

warp the current/previous frame to compensate camera motion;

compute residual absolute difference;

summarize residuals spatially.

Possible features:

global residual mean/max

3×3 or 5×5 spatial grid statistics

residual motion magnitude

left/right asymmetry

center/road-region residual

temporal max/mean between sampled positions.

Call the resulting representation something like:

residual_motion_features.

Compare:

M0

current M_motion only.

M1

global motion + residual motion concatenated.

M2

residual motion only.

M3

current M_motion network unchanged, but residual motion used only as a late local event cue.

Do not use residual motion in final decoding unless validated out-of-fold.

Main questions:

does residual motion improve ENTRY more than current global motion?

does it improve COLLISION without duplicating camera-shift fusion?

is it complementary to PH_phase?

Experiment 4 — Event-boundary / change-point supervision

Hypothesis:

The most useful information from phase supervision may be its two transitions rather than its three region labels.

Add lightweight boundary heads:

b_entry(t)
b_collision(t)

from the final temporal hidden representation.

Also test using temporal feature differences:

Δh_t = h_t - h_{t-1}

and optionally:

h_{t+1} - h_t.

Possible head inputs:

BND1

h_t

BND2

concat(h_t, Δh_t)

BND3

local [h_{t-1}, h_t, h_{t+1}] projection.

Targets should be narrow Gaussian/soft peaks around the sampled ENTRY and COLLISION positions.

Do not use seconds or FPS to define the target width.

Test widths such as:

1 sampled position

2 sampled positions.

Initially use boundary heads only as auxiliary training supervision.

Then separately test whether local boundary scores can rerank top-K direct candidates.

Do NOT perform whole-video structured decoding.

Main metrics:

ENTRY/COLLISION hit

normalized MAE

Recall@1/2/4/8

catastrophic-error rate.

Experiment 5 — Complementarity analysis

Do not immediately combine all successful features.

For every candidate auxiliary signal, measure its value in two ways.

A. Standalone value

Compare:

control
vs
control + auxiliary signal.

B. Ensemble complementarity

Keep the current v5 non-candidate members fixed.

For example:

C0 + X_ema + M_motion + PH

versus:

C0 + X_ema + M_motion + PH + RISK

and corresponding same-size controls where necessary.

If ensemble size changes, also include a same-size control made by adding another seed/model of an existing family.

This is important: an improvement caused only by adding one more model should not be attributed to the auxiliary signal.

Measure prediction diversity between the candidate model and existing ensemble members:

disagreement rate

per-video error correlation

ENTRY error correlation

COLLISION error correlation

oracle ensemble gain.

A useful auxiliary feature may have only modest standalone score but still be valuable if its errors are complementary.

Experiment 6 — Pairwise combinations

Only after single-signal experiments are finished, combine the strongest two candidates.

Likely combinations to test if individually positive:

PH + risk progression

PH + boundary supervision

PH + lane intrusion

motion + residual motion

PH + risk + motion, only if justified.

Do not combine features that independently failed.

Avoid a combinatorial sweep.

Evaluation protocol

Use staged evaluation.

Stage A — screening

Fixed 279/70 split, seed 0.

Reject clearly bad ideas quickly.

Stage B — validation

Run 5-fold CV for every candidate that survives Stage A.

Stage C — robustness

Run multiple seeds for the best 2–3 methods.

Use the same seeds/folds as the current LC-v2 experiments whenever possible.

Report:

overall Stage-2 score

ENTRY hit

COLLISION hit

side F1

evasion F1

NEXAR score

non-NEXAR score



1000-frame score

ENTRY normalized MAE

COLLISION normalized MAE

catastrophic errors

Recall@1/2/4/8

runtime / parameter increase.

For all paired comparisons, report:

mean fold improvement

number of folds improved

per-video paired delta

bootstrap confidence interval or another paired uncertainty estimate.

Success criteria

A candidate feature should be considered genuinely useful if it satisfies most of the following:

improves 5-fold OOF overall score by at least approximately +0.01;

improves at least 4/5 folds;

improves the metric it is intended to target:

lane intrusion -> ENTRY

motion/risk -> mainly COLLISION

boundary -> event precision;

does not hurt >1000-frame score by more than 0.01;

gives positive incremental gain when added to the existing v5 ensemble;

beats a same-size ensemble control;

remains strictly FPS/duration/timestamp blind.

A candidate can still be valuable with a smaller standalone gain if its ensemble complementarity is unusually strong.

Priority order

Run in this order:

collision-risk progression supervision

event-boundary/change-point supervision

ego-motion-compensated residual motion

victim-lane intrusion supervision.

Risk and boundary experiments are cheapest and should be completed first.

Lane intrusion may require substantial pseudo-label infrastructure, so do not spend major compute on it until pseudo-label quality is validated.

Required final report

Produce one report summarizing:

which auxiliary signal gives the largest standalone gain;

which gives the largest ensemble gain;

which mainly improves ENTRY;

which mainly improves COLLISION;

whether risk supervision complements or duplicates PH_phase;

whether residual motion complements or duplicates M_motion;

whether explicit boundary supervision explains part of PH_phase's benefit;

whether lane-intrusion supervision provides a genuinely ENTRY-specific gain;

which improvements survive 5-fold CV and multiple seeds;

the best evidence-supported combination for the next Stage-2 model.

Do not build a leaderboard submission simply because a fixed-split result is high.

Only recommend a new submission when the improvement survives 5-fold CV and improves the current v5-style ensemble under a matched comparison.