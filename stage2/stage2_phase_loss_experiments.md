Objective

This experiment is specifically about validating phase supervision and finding the best way to utilize phase loss.

Do NOT treat this as a general Stage-2 architecture search.

Keep fixed:

geometry-tuned DINOv3-S backbone

DINO feature cache

temporal sampling

LCPyramid backbone

motion features

side/evasion heads

optimizer/training recipe unless explicitly varied

existing camera-motion collision fusion

FPS-blind constraint.

Do not introduce a new backbone, Transformer, SSM, optical-flow model, detector, or unrelated auxiliary objective.

The questions to answer are:

Does semantic PRE/BETWEEN/POST supervision itself improve Stage 2?

What representation of phase supervision works best?

Where and with what weight should phase loss supervise the temporal network?

After training with phase loss, should phase predictions be used at inference, or is phase best used only as an auxiliary training signal?

The primary success criterion is not a predetermined absolute score.

A winning phase method should:

improve 5-fold OOF score by >= +0.01 versus its exactly matched no-phase control;

improve at least 4/5 folds;

not reduce the >1000-frame score by more than 0.01;

improve the current v5 ensemble when replacing the existing PH_phase members.

Use multiple seeds for finalists and report uncertainty.

Critical confound to fix first

Current phase_loss.py changes more than phase supervision:

Current PH approximately uses:

1.0 L_phase + 0.75 L_transition + 0.25 L_direct + 0.05 L_mono + 0.5 L_side + 0.5 L_evasion

while the no-phase control uses:

1.0 L_direct + 0.5 L_side + 0.5 L_evasion.

Therefore existing PH-vs-NT results do NOT cleanly isolate phase supervision because the direct-event loss weight also changes from 1.0 to 0.25.

The first experiment must disentangle this.

Create matched controls:

A0:
1.0 L_direct + attributes
No phase.

A1:
0.25 L_direct + attributes
No phase.

A2:
1.0 L_direct + λ L_phase + attributes

A3:
0.25 L_direct + λ L_phase + attributes

Then add transition/monotonic terms separately.

This should determine whether the gain comes from:

semantic phase supervision;

reduced direct-event weighting;

or their interaction.

Experiment 1 — Verify that phase semantics matter

Use the same model and training setup.

Compare:

no phase supervision;

correct GT phase supervision;

shuffled-phase control.

For the shuffled control, preserve approximately the same PRE/BETWEEN/POST class frequencies but destroy the semantic alignment between the true ENTRY/COLLISION and phase transitions. For example, use phase boundaries from another training video or valid randomized transitions.

Do not use shuffled labels in validation.

If true phase supervision improves substantially more than shuffled phase supervision, this is evidence that accident progression semantics are responsible rather than generic auxiliary-task regularization.

Also optionally test a phase head attached to h.detach() as a diagnostic. It should learn phase without sending phase gradients into the shared temporal representation. Compare this with normal phase supervision to determine whether the gain actually comes from representation shaping.

Experiment 2 — Find the best phase representation

Hold all non-phase losses and weights fixed.

Test three phase representations.

PH-CAT — Current categorical phase

Three-way prediction:

PRE / BETWEEN / POST

using frame-wise cross entropy.

This is the reference.

PH-ORD — Ordinal/cumulative phase

Exploit the known ordering:

PRE -> BETWEEN -> POST.

Predict two logits:

q_entry(t) = P(phase >= BETWEEN)
q_collision(t) = P(phase >= POST).

Targets:

y_entry(t) = 1[t >= ENTRY]
y_collision(t) = 1[t >= COLLISION].

Use BCE or an ordinal logistic formulation.

Enforce or regularize:

q_entry(t) >= q_collision(t).

The implied phase probabilities are:

P(PRE) = 1-q_entry
P(BETWEEN) = q_entry-q_collision
P(POST) = q_collision.

This remains a phase-supervision experiment; do not replace the direct ENTRY/COLLISION spotting heads.

PH-SOFT — uncertainty-aware categorical/ordinal phase

The exact sampled location of a phase transition can be ambiguous because Stage 2 temporally subsamples frames.

Instead of making the phase target change discontinuously at exactly one sampled position, test soft labels over a very small neighborhood around ENTRY and COLLISION.

Test only compact widths such as:

1 sampled position

2 sampled positions.

Do not define widths in seconds or use FPS.

Compare PH-CAT, PH-ORD and PH-SOFT under otherwise identical conditions.

Experiment 3 — Determine which components of phase loss actually help

For the best representation from Experiment 2, perform controlled loss ablations.

Starting from:

L = L_direct + λ_phase L_phase + attributes

evaluate adding individually:

+ L_transition

and

+ L_monotonic.

Then combine them only if each is beneficial.

Do not immediately reproduce the existing composite loss.

Test a small phase-weight sweep:

λ_phase = 0.1, 0.25, 0.5, 1.0, 2.0.

Use the same direct-event loss weight across this sweep.

For transition and monotonic terms, use a compact weight sweep only around values supported by the initial results.

Report whether too-strong phase supervision causes negative transfer to precise ENTRY/COLLISION spotting.

Also measure gradient interaction on several training batches:

cosine similarity between gradients from:

L_direct

L_phase

L_transition

on the shared temporal-pyramid parameters.

This is diagnostic only, but use it to determine whether phase and direct spotting objectives are cooperative or conflicting at different λ values.

Experiment 4 — Where should phase supervision be attached?

Using only the best phase target/loss from Experiments 1-3, compare:

S1: final-level auxiliary phase head

Current approach:
phase head from final full-resolution fused hidden sequence.

S2: multi-scale phase supervision

Attach lightweight phase heads to multiple temporal-pyramid levels.

Downsample phase targets appropriately to each temporal level and average/weight their phase losses.

The inference model does not need these auxiliary heads if they are only used for training.

The purpose is to test whether phase progression is a useful representation constraint at coarse temporal scales as well as the final fine level.

S3: phase-conditioned event head

Allow the ENTRY/COLLISION event head to receive phase representation in addition to hidden features.

Test both:

detached phase probabilities;

normal end-to-end phase probabilities.

Keep this lightweight, e.g. small projection/concatenation rather than another temporal model.

This tests whether phase should merely regularize the representation or explicitly inform event spotting.

Experiment 5 — Determine the best inference-time use of phase

Use the SAME trained checkpoint for these comparisons whenever possible.

Compare:

D0 — Auxiliary only

Ignore phase predictions at inference.

Decode ENTRY/COLLISION exactly as current direct decoding.

This is the primary reference.

D1 — Phase transition decoding

Derive ENTRY from the strongest PRE->BETWEEN transition and COLLISION from the strongest BETWEEN->POST transition.

This assesses whether phase could itself serve as the event predictor.

D2 — Local phase-assisted reranking

Start from the top-K candidates of the direct ENTRY/COLLISION heads.

Use phase evidence only locally around those candidates.

For ENTRY, use local evidence for:

PRE -> BETWEEN.

For COLLISION:

BETWEEN -> POST.

Do not accumulate phase likelihood across the entire video.

Test K around 4/8 only.

D3 — Existing global structured decoder

Do not spend substantial compute re-optimizing this. Use the existing result as the negative/control result showing that whole-video summed phase decoding is harmful.

The goal is to determine whether phase is best used:

only during training;

for local refinement;

or directly for event prediction.

Evaluation strategy

Use a staged process.

Stage A:
fixed 279/70 split, seed 0, for rapid screening and debugging.

Stage B:
5-fold CV for all phase formulations that survive Stage A.

Stage C:
multiple seeds only for the best 2-3 formulations.

For every experiment report:

overall Stage-2 score

ENTRY hit

COLLISION hit

side F1

evasion F1

NEXAR score

non-NEXAR score



1000-frame score

normalized ENTRY/COLLISION MAE

catastrophic event-selection rate

phase frame accuracy / phase macro-F1

transition localization error

train-vs-validation gap.

Also perform paired bootstrap or another paired per-video analysis between the best phase model and its matched no-phase control. Report whether the gain is broad or caused by a small number of clips.

Important: FPS may be used only after prediction for the official ±0.3 s evaluation. The model/loss/sampling/decoder must remain FPS-, duration-, and timestamp-blind.

Final ensemble test

Only after finding the best phase utilization method, test its actual value in v5.

Keep these current v5 families unchanged:

C0 + X_ema + M_motion

Then compare, with the same ensemble size:

+ NT/no-phase control

+ current PH_phase

+ best new phase method.

Do not change motion fusion or other v5 components during this comparison.

This is the key practical test: does the improved phase model provide complementary errors to the existing ensemble?

Required conclusions

The final report must explicitly answer:

Is the benefit genuinely caused by semantic phase supervision?

Was any previous gain actually caused by changing the direct-loss weight?

Categorical vs ordinal vs soft phase: which representation is best?

What phase-loss weight is best?

Are transition and monotonic terms useful independently?

Does multi-scale phase supervision help?

Should phase be auxiliary-only, event-conditioned, or used in decoding?

Does phase mainly improve ENTRY, COLLISION, or both?

Does it specifically reduce catastrophic errors on long videos?

Does the best phase formulation improve the full v5 ensemble?

Do not build a new submission until these questions are answered.

The purpose of this study is to understand how phase supervision should be utilized, then use that evidence to design the next Stage-2 submission.