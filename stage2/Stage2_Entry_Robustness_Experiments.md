First, read the recent Stage 2 experiment reports, code, configs, and saved results carefully. In particular inspect the lineage and differences between v8, v10, v12, and v13, including:

reports/stage2_robustness_campaign.md

reports/stage2_v13_robustness.md

reports/stage2_generalization_research.md

reports/stage2_phase_supervision_study.md

reports/stage2_complementary_signals_interim.md

stage2/generalization/

stage2/aux_signal_experiments/

submission_tools/v8_stage2/

submission_tools/v10_stage2/

submission_tools/v12_stage2/

Do not trust this prompt blindly. Verify every architectural and experimental detail from the repository before implementing anything.

Main observation

The current best real Stage-2 leaderboard model is v8, LB ≈ 0.5929.

Later models such as v10/v12 obtain better normal validation and much better frame-rate robustness, but score lower on the real leaderboard.

The remaining weakness appears concentrated in ENTRY localization, especially when the ENTRY→COLLISION gap is long.

Important existing observations include approximately:

v8 ENTRY accuracy for gap 1.5–2.5 s: ~0.42

v12: ~0.33

v8 long-gap ENTRY bias: roughly +0.3–0.4 s late

newer models tend to predict ENTRY too close to COLLISION

newer models also show stronger temporal-position priors under crop/truncation tests

frame-rate robustness is already considerably improved after v8

gap-balanced loss, gap prior decoding, pre-collision truncation, causal ENTRY, larger ViT-B backbone, ordinary object tracking, and simple decoding changes have already been tested and did not reliably solve long-gap ENTRY

object-appearance features were one of the only interventions that actually moved the long-gap ENTRY result in the correct direction

BND2-style boundary supervision [h_t, h_t-h_{t-1}] was also useful in previous experiments

The working hypothesis is therefore:

The current model often detects the point where the intruding vehicle becomes visually obvious, rather than the earlier onset of the lane-entry maneuver. Temporal pooling, low spatial resolution, pseudo-labelled short-gap data, and excessive cross-frame-rate invariance may suppress subtle early ENTRY evidence.

The goal of this experiment campaign is not to maximize ordinary CV score.

The goal is:

Build an ENTRY predictor that is genuinely driven by video/object/lane geometry and improves long-gap ENTRY localization while preserving v8's frame-rate robustness and overall performance.

Experiment 0 — Build a better diagnostic benchmark

Before modifying the architecture, create a rigorous ENTRY robustness evaluation suite.

In addition to current CV metrics, report:

ENTRY accuracy by ENTRY→COLLISION gap:

<0.5 s

0.5–1.0 s

1.0–1.5 s

1.5–2.5 s



2.5 s

Signed ENTRY timing bias in seconds for long-gap samples.

ENTRY performance on:

native rate

1/2 rate

1/3 rate

crop 0.5

crop 0.25

Predicted-vs-GT ENTRY relative-position slope under temporal cropping.

Source-wise results.

LOSO where feasible.

A gap-OOD evaluation:
create folds where the distribution of ENTRY→COLLISION gaps differs substantially between train and validation.
For example:

train mostly short/medium gaps, validate long gaps

train mostly long/medium gaps, validate short gaps

Do not use FPS as an inference input. It may only be used for offline evaluation.

Also create a simple statistical-prior baseline which predicts ENTRY using only things such as normalized temporal position, clip length/frame count, and optionally predicted COLLISION position.

This is a diagnostic baseline only.

We want to measure how much the neural model actually improves over a timing-prior model.

Experiment 1 — v8 + fine-resolution boundary ENTRY head

Use v8 as the main baseline, not v12.

Keep the existing v8 COLLISION and attribute pathways unchanged.

Add a new ENTRY-specific boundary branch before the strongest temporal pooling destroys fine temporal information.

Candidate design:

fine/full-resolution hidden h_t
→ concatenate [h_t, h_t - h_(t-1)]
→ small boundary MLP / temporal conv
→ entry_boundary_logits

Then fuse with the original v8 ENTRY logits:

entry_final = entry_v8 + alpha * entry_boundary

Test a small number of α values or learn a restrained residual gate.

The new branch should act as a residual correction, not completely replace the v8 predictor.

Train it using manually labelled Stage-2 samples only initially.

Do NOT use MM-AU/CCD pseudo ENTRY labels in this first experiment.

Compare:

v8 baseline

v8 + BND2-style ENTRY branch

boundary branch alone

v8 + boundary branch ensemble/fusion

Primary question:

Can explicit local feature change improve maneuver-onset detection, especially at 1.5–2.5 s gaps?

Experiment 2 — Object + ego-lane geometry ENTRY branch

This is the highest-priority architectural experiment.

Existing object appearance features gave one of the first real improvements on long-gap ENTRY. Extend this into an explicitly physical representation.

Use existing detector/tracker infrastructure if reliable.

For several candidate vehicles per frame, obtain features such as:

bounding box position and scale

box growth rate

lateral velocity after ego-camera motion compensation

object appearance embedding

object appearance temporal change

vehicle bottom/contact point

ego-lane boundaries / ego-lane corridor

signed lateral distance from vehicle contact point to the relevant ego-lane boundary

derivative of this lane-boundary distance

fraction of vehicle occupying the ego-lane corridor

derivative of lane overlap

heading/orientation change if obtainable reliably

presence/tracking confidence

The objective should be to detect:

the onset of lane intrusion, not merely proximity to collision.

Construct a lightweight per-object temporal encoder.

For each tracked object:

object geometry + lane-relative geometry + appearance
→ short temporal model
→ per-frame entry_onset_score

Then aggregate across candidate objects using max/attention.

Fuse this as a residual with the original v8 ENTRY predictor.

Keep the original v8 COLLISION predictor unchanged initially.

Test at least:

A. v8 baseline
B. v8 + previous object-appearance features
C. v8 + lane-relative geometry only
D. v8 + object appearance + lane geometry
E. v8 + object/lane branch + BND2 boundary branch

Do not make the architecture unnecessarily large. Dataset size is small and overfitting has historically been a major problem.

Experiment 3 — Preserve native-resolution ENTRY evidence

The current robust models use aggressive stride augmentation and cross-rate consistency.

Test whether this is hurting the subtle ENTRY boundary.

Start from v8.

Compare:

A. original v8 stride probabilities
0.50 / 0.25 / 0.25 for stride 1/2/3

B. equal stride mix

C. original v8 mix + consistency only for COLLISION

D. original v8 mix + weak ENTRY consistency

E. equal mix + consistency only for COLLISION

The hypothesis is:

COLLISION should be highly invariant across temporal sampling rates, while early ENTRY evidence may legitimately disappear at low temporal resolution.

Therefore do not force the high-resolution ENTRY distribution to exactly match stride-3 predictions.

For ENTRY, optionally test a broad consistency target, e.g. consistency between smoothed distributions or coarse temporal bins rather than exact localization.

Measure whether removing strong ENTRY consistency improves long-gap localization while retaining the low-FPS gains.

Experiment 4 — Counterfactual gap augmentation

Previous gap-balanced loss did not work because it only reweights examples.

Instead create examples that explicitly weaken the correlation between ENTRY appearance and ENTRY→COLLISION gap.

For suitable clips, perform temporal transformations of the BETWEEN region.

Examples:

stretch the frames between ENTRY and COLLISION

compress the BETWEEN interval

repeat/interpolate selected BETWEEN frames

vary the duration between the same visual ENTRY onset and collision

Transform all labels consistently.

The important property is:

visually equivalent ENTRY onset cues should occur with multiple different ENTRY→COLLISION gaps.

Do not create unrealistic optical artifacts if avoidable.

Test conservative transforms first.

Compare:

v8 baseline

v8 + gap balancing

v8 + counterfactual gap augmentation

v8 + object/lane ENTRY branch + counterfactual gap augmentation

Measure whether the ENTRY-vs-gap performance slope becomes less negative.

Experiment 5 — Remove pseudo-label influence from ENTRY

Investigate whether MM-AU/CCD pseudo-labelled ENTRY supervision caused the v10/v12 shortcut.

Create controlled matched experiments.

Use additional MM-AU/CCD examples in several ways:

A. no extras — v8-style reference

B. extras supervise COLLISION only

C. extras supervise COLLISION + ENTRY as in v10/v12

D. extras supervise COLLISION while their gradients are prevented from updating the ENTRY-specific branch

E. if feasible, extras provide very low-weight ENTRY supervision only when a geometry-based confidence criterion is extremely high

Keep all other conditions matched.

The important comparison is:

COLLISION-only extras vs COLLISION+teacher-ENTRY extras.

Report long-gap ENTRY, position-prior metrics, gap-OOD metrics, and real-source validation separately.

Do not declare pseudo-label expansion beneficial based only on aggregate CV.

Experiment 6 — Hard-boundary supervision

Create explicit local supervision around GT ENTRY.

For each labelled ENTRY, define temporal states such as:

clearly BEFORE ENTRY

onset / boundary neighborhood

AFTER ENTRY but before obvious full intrusion

Use narrow local windows.

Potential objectives:

boundary BCE/CE

contrastive feature loss between BEFORE and AFTER

ranking constraint:
score(entry) > score(entry - δ)
and
score(entry) > score(entry + δ) where appropriate

temporal derivative supervision

The purpose is to teach the model the transition itself, not the global phase.

Avoid repeating the previously harmful whole-clip structured phase decoder.

Phase/boundary losses may be used for representation training, but final inference should remain evidence-driven and primarily use direct ENTRY logits unless experiments clearly prove otherwise.

Selection criteria

Do not select models primarily by aggregate CV.

A candidate should only be considered genuinely better than v8 if it satisfies most of:

overall duplicate-clean CV approximately preserved or improved

1/3-rate robustness approximately preserved

ENTRY 1.5–2.5 s clearly exceeds v8's ~0.42 baseline or shows a statistically meaningful improvement

long-gap ENTRY late bias decreases below the v8 level

crop-0.25 ENTRY is at least comparable to v8 (~0.53)

predicted-vs-GT ENTRY position slope under crop does not regress

gap-OOD validation improves

improvements appear across multiple seeds

improvements are not restricted to MMAU or another single source

the model clearly outperforms the statistical-prior baseline on long-gap and position-shift tests

Track short-gap performance too, but do not sacrifice long-gap behavior merely to obtain nearly perfect performance on <0.5 s gaps.

Experiment order

Do not run every combination immediately.

Use this priority:

diagnostic / gap-OOD benchmark

v8 + BND2-style fine ENTRY boundary branch

v8 + object/lane geometry branch

ENTRY-specific consistency ablation

counterfactual gap augmentation

pseudo-label isolation experiment

combine only the interventions that individually show complementary gains

Use small screening runs first, then ≥3–4 seeds for promising candidates.

Avoid interpreting 1–2 lucky seeds as a result.

Final report

Produce a concise but rigorous report containing:

exact experiment configs

architecture diagrams/descriptions

parameter counts

training settings

all important validation metrics

gap-wise ENTRY results

signed ENTRY timing errors

crop-position robustness

frame-rate robustness

source-wise results

gap-OOD results

comparison against v8

comparison against statistical-prior baseline

per-seed variance

which interventions genuinely improve visual grounding vs merely improve ordinary CV

Most importantly, answer these questions:

Why does v8 transfer better to the real leaderboard than v10/v12?

Is the long-gap failure caused mainly by training-gap priors, loss design, temporal pooling, spatial resolution, or missing object/lane evidence?

Can we make ENTRY depend more strongly on observable lane-intrusion geometry?

Can we improve long-gap ENTRY without losing the frame-rate robustness gained by later models?

What is the strongest evidence-backed candidate for the next leaderboard submission?

Do not build or submit a new leaderboard package merely because it has higher normal CV. A candidate must first demonstrate improvement on the ENTRY robustness diagnostics above.