

Focus only on Stage 3.

Goal

Conduct a controlled experimental study to determine:

Which additional video-derived geometric features improve the current Stage 3 methodology beyond the geometry already used by the existing model?

Do not assume the candidates in this prompt are correct.

First inspect the latest repository state, especially:

stage3/ARCHITECTURE.md

stage3/README.md

stage3/geometry/

stage3/flow/

stage3/model/

stage3/trainer/

Stage 3 configs

recent Stage 3 experiment reports/results if available

Understand exactly what geometry/motion information is already present before proposing or implementing anything.

The current method already contains substantial geometry information such as SEA-RAFT optical flow, rotational-flow removal, FOE, radial expansion, multi-lag rho estimates, motion confidence, speed/height proxy, pitch proxy, and temporal modeling.

Therefore:

Do not add redundant features simply because they sound useful.

The objective is to find geometric signals that provide genuinely new information and improve validation performance.

Experimental constraints

For fast iteration, all experiments must use:

BATON only

current BATON train/validation split

20 epochs

same preprocessing

same optimizer and training schedule as the current best baseline unless a minimal feature-specific change is required

same initial random seed

same model capacity wherever possible

same evaluation and decoding protocol

Do NOT add:

ADAS

DriveDNA

additional external datasets

during this study.

Do not perform a large architecture search.

This is a:

geometry feature attribution / ablation study

not a general Stage 3 hyperparameter search.

CRITICAL: Fixed evaluation protocol

All geometry experiments must be evaluated using the current best decoding thresholds:

Acceleration threshold

Use:

[
\boxed{0.4}
]

i.e. preserve the current best acceleration decoding behavior corresponding to the ±0.4 threshold.

Do not independently optimize the acceleration threshold for G1, G2, G3, etc.

Steering threshold

Use:

[
\boxed{7^\circ}
]

for LEFT / STRAIGHT / RIGHT decoding.

Again, do not optimize steering thresholds separately for individual geometry experiments.

The purpose is to measure whether the geometry itself improves the model, not whether threshold tuning improves the validation score.

If the existing repository contains slightly different threshold semantics, inspect the implementation first and reproduce the exact current-best decoding rule associated with:

accel threshold = 0.4

steer threshold = 7

rather than blindly changing constants.

Competition evaluation metric

Every main experiment must be compared using the actual Stage 3 competition evaluation methodology.

Calculate:

Acceleration Macro-F1

Steering Macro-F1

Stage 3 competition score

using:

0.7F1_{\mathrm{accel}}
+
0.3F1_{\mathrm{steer}}
}
]

This competition score must be the primary model-selection metric.

Do not rank geometry experiments using regression loss alone.

STOPPED handling must match competition evaluation

Be especially careful about STOPPED.

Use the current competition-compatible handling:

Acceleration

STOPPED remains one of the acceleration categories and therefore participates normally in acceleration Macro-F1.

Acceleration evaluation should include all acceleration classes according to the current competition implementation.

Steering

Frames whose ground-truth acceleration state is STOPPED must be excluded from steering Macro-F1 evaluation.

However:

the model must still output steer_label for every frame;

STOPPED frames are not deleted from inference;

steering predictions on GT-STOPPED frames simply do not contribute to steering Macro-F1.

Do NOT accidentally:

evaluate steering on STOPPED frames;

remove STOPPED frames before running the overall sequence decoder;

use predicted STOPPED status instead of the competition's proper ground-truth STOPPED evaluation mask when calculating validation steering F1;

skip steering outputs for STOPPED frames.

Before beginning experiments, verify that the repository's validation/evaluation implementation follows this exactly.

If necessary, fix the evaluation implementation before running the study, but do not otherwise alter the baseline.

Step 1 — Establish exact baseline

Reproduce the current strongest Stage 3 BATON configuration for exactly 20 epochs.

Use the fixed decoder:

Acceleration threshold: 0.4
Steering threshold:     7 degrees

and the competition-compatible STOPPED handling described above.

Call this:

G0_baseline

Record:

best validation competition score

acceleration Macro-F1

steering Macro-F1

best epoch

regression metrics/losses for:

longitudinal acceleration

speed

steering angle

yaw rate if available

parameter count

training time

inference overhead if relevant

This G0 result is the reference point for every later experiment.

Do not change thresholds between G0 and geometry experiments.

Step 2 — Audit existing geometry

Before implementing new features, create a concise table describing every current Stage 3 geometric feature.

For every feature record:

name

mathematical / physical meaning

where it is computed

whether it is spatial or scalar

whether it mainly informs acceleration, steering, or both

whether it is metric or scale-invariant

likely failure modes

whether one of the candidates below would duplicate it

Pay particular attention to existing:

SEA-RAFT flow

rotational flow / angular velocity

rotation confidence/residual

FOE

radial expansion

rho k=2 / k=4

rho confidence/residual

speed/height proxy

near/mid/far expansion

pitch proxy

motion magnitude

still fraction

confidence maps

temporal TCN features

This audit should decide whether each proposed experiment is actually justified.

Step 3 — Test additional geometry candidates

Prioritize small, interpretable geometry additions rather than larger learned backbones.

G1 — Better static / rigid-scene mask

Investigate whether the current static_valid truly identifies stationary scene structure or mostly represents flow-confidence / border validity.

Build a better rigidity mask if justified.

Possible sources:

global-motion residual

RANSAC rigidity

epipolar consistency

estimated ego-motion consistency

lightweight semantic rejection of vehicles/pedestrians

road/background-biased masking

The objective is to prevent independently moving objects from contaminating estimates of:

rotation

FOE

expansion

rho

visual speed

Prefer a geometry-based lightweight approach before adding a heavy segmentation network.

Experiment:

G1_rigid_mask

G2 — Explicit visual ego-speed geometry

The current model estimates quantities related to:

[
\rho \approx \frac{a}{v}
]

and already contains a speed/height proxy.

Investigate whether a stronger video-derived estimate of ego speed supplies information missing from the current representation.

Possible cues:

road-plane optical-flow geometry

ground-plane translation

lower-image perspective flow

road/lane scale cues

temporally smoothed visual speed

camera-height-normalized speed proxies

Do not force absolute metric speed if the monocular evidence cannot support it reliably.

Relative or normalized speed is acceptable if it improves generalization.

Test derived quantities such as:

[
\hat v_t
]

[
\Delta\hat v_t
]

and especially:

\rho_{2,t}\hat v_t
]

\rho_{4,t}\hat v_t.
]

Experiment:

G2_visual_speed

If computationally cheap, separate:

G2a_speed_only

G2b_rho_times_speed

G2c_speed_plus_rho_times_speed

to determine what actually contributes.

G3 — Road / lane geometry

Investigate explicit road-relative geometry.

Potential features:

[
e_y = \text{lane-center offset}
]

[
\psi = \text{lane-heading error}
]

[
\kappa = \text{road curvature}
]

as well as:

vanishing-point x/y

road horizon

lane direction

lane curvature

curvature derivative

heading derivative

drivable-area geometry

Main hypothesis:

these quantities expose steering geometry that generic optical flow does not represent explicitly.

Keep the implementation lightweight.

Experiment:

G3_road_lane_geometry

Analyze acceleration and steering separately because this feature may predominantly affect steering F1.

G4 — Road-relative pitch dynamics

The current system already has a pitch proxy.

First determine exactly what information it captures.

Only proceed if road-relative pitch provides genuinely new information.

Try to distinguish:

camera/body pitch
        -
road slope / road horizon

to obtain:

[
\theta_{\mathrm{rel}}
]

and possibly:

[
\dot\theta_{\mathrm{rel}},
\qquad
\ddot\theta_{\mathrm{rel}}.
]

Hypothesis:

Braking and acceleration cause suspension/body pitch transients, but raw camera pitch is contaminated by road slope. Road-relative pitch may therefore produce a cleaner longitudinal dynamics cue.

Experiment:

G4_road_relative_pitch

Skip this experiment if the audit shows that the existing representation already captures essentially the same quantity.

G5 — Curvature and lateral-dynamics geometry

Investigate physically meaningful combinations of estimated visual speed and yaw:

[
\kappa_t =
\frac{\omega_{z,t}}
{\hat v_t+\epsilon}
]

and:

\hat v_t \omega_{z,t}.
]

These may provide useful intermediate variables for steering.

Where BATON contains corresponding signals, auxiliary supervision is allowed.

However, CAN-derived values may only be used as training supervision, never as DACON inference inputs.

Experiment:

G5_curvature_lateral

Step 4 — Controlled combinations

Do NOT combine all features immediately.

First complete the individual experiments:

G0 baseline
G1 rigid mask
G2 visual speed / rho×v
G3 road/lane geometry
G4 road-relative pitch
G5 curvature/lateral

Compare them under the exact same:

20 epochs
BATON split
accel threshold = 0.4
steer threshold = 7
competition metric
STOPPED evaluation rule

Then combine only individually useful and complementary features.

Possible combination experiments:

G6 — Best acceleration geometry

Combine only geometry features that demonstrably improve acceleration F1.

G7 — Best steering geometry

Combine only geometry features that demonstrably improve steering F1.

G8 — Best complementary combination

Combine the strongest non-redundant acceleration and steering geometry.

Do not include features that failed individually unless there is a strong physical interaction hypothesis.

Anti-leakage requirement

Every feature available to the Stage 3 model at DACON inference must be computable from the video.

BATON signals such as:

vEgo

aEgo

steering angle

yaw rate

may be used as:

regression targets

auxiliary targets

diagnostic ground truth

but must not be inference inputs.

For example, it is valid to train:

[
\text{video}\rightarrow\hat v
]

using CAN vEgo.

It is NOT valid to supply vEgo directly to the final Stage 3 model.

Architecture philosophy

Prefer:

video
  ↓
SEA-RAFT / lightweight geometry estimators
  ↓
interpretable geometry
  ↓
existing Stage 3 motion + physics representation
  ↓
existing temporal model

over:

video
  ↓
much larger generic RGB backbone
  ↓
larger temporal model

The purpose is to identify a Stage-3 analogue of the large Stage-2 gain obtained from camera-motion information:

a compact feature that exposes target-relevant physical information and generalizes better than simply increasing network capacity.

When adding scalar features, prefer integrating them into the existing physics branch.

When spatial structure matters, add spatial maps to the motion branch.

Keep each modification as small as possible so feature attribution remains meaningful.

Diagnostics

Do not rely solely on final competition F1.

For each geometry variable, measure its relationship to BATON ground truth where possible.

Examples:

Visual speed

Compare:

[
\hat v
\quad\text{vs}\quad
v_{\mathrm{Ego}}.
]

Report:

correlation

MAE/RMSE if scale is meaningful

error by speed range

Geometry-derived acceleration

Compare:

[
\rho\hat v
\quad\text{vs}\quad
a_{\mathrm{Ego}}.
]

Check whether it correctly captures:

acceleration sign

transition timing

magnitude ordering

Visual yaw / curvature

Compare against available CAN/vehicle-dynamics signals.

Road-relative pitch

Measure correlation with longitudinal acceleration and determine whether the relationship remains after separating road slope.

Per-class analysis

Report F1 and confusion matrices for:

Acceleration

DECELERATING

CONSTANT

ACCELERATING

STOPPED

Pay particular attention to:

CONSTANT ↔ ACCELERATING
CONSTANT ↔ DECELERATING
STOPPED ↔ CONSTANT

Steering

Evaluate only according to the competition-valid driving frames, excluding GT STOPPED frames.

Analyze:

LEFT

STRAIGHT

RIGHT

Pay particular attention to:

STRAIGHT ↔ weak LEFT
STRAIGHT ↔ weak RIGHT

These diagnostics should help determine which geometric cue actually solves which error mode.

Required experiment table

Produce a table of the form:

ID

Geometry

Accel F1

Steer F1

Competition Score

Δ vs G0

Best Epoch

Compute

G0

Existing baseline







0





G1

Rigid/static mask













G2

Visual speed / rho×v













G3

Road/lane geometry













G4

Road-relative pitch













G5

Curvature/lateral













G6

Best acceleration combination













G7

Best steering combination













G8

Best combined geometry













Every value in this table must use:

Acceleration threshold = 0.4
Steering threshold     = 7 degrees
GT STOPPED excluded from steering Macro-F1
Competition Score      = 0.7 × Accel Macro-F1
                       + 0.3 × Steer Macro-F1

Do not insert per-experiment threshold-tuned results into this main table.

If you optionally perform threshold sweeps for diagnostic purposes, report them separately and clearly mark them as diagnostic only.

Evaluation discipline

The primary comparison is:

S_{\mathrm{geometry}}

S_{\mathrm{G0}}
}
]

under identical decoding and evaluation.

Rank experiments primarily by:

Stage 3 validation competition score

acceleration Macro-F1

steering Macro-F1

per-class robustness

route/clip robustness

physical correctness

additional inference cost

Do not claim an experiment works because:

training loss decreased;

regression MAE decreased but competition F1 did not;

its individually optimized threshold is better;

training score increased.

Because these are fast 20-epoch BATON-only experiments, small score differences may be random noise.

Explicitly classify results as:

clear improvement

promising but uncertain

neutral

harmful

instead of over-interpreting tiny differences.

Deliverables

At the end, create a concise Markdown report containing:

exact repository commit tested

exact G0 baseline

audit of existing Stage 3 geometry

evaluation implementation verification

confirmation that:

accel threshold = 0.4

steer threshold = 7

STOPPED handling matches competition rules

hypothesis for every candidate

implementation details

experiment table

acceleration per-class results

steering per-class results

geometry-vs-CAN diagnostics

failure cases

features that genuinely add information

features that were redundant with existing Stage 3 geometry

compute/inference overhead

smallest recommended geometry configuration for the next full training run

The final research question is:

Which video-derived geometric variables contain complementary information about ego longitudinal acceleration and steering that the existing SEA-RAFT + rotation + FOE + expansion + rho + physics-TCN system does not already capture, when every candidate is evaluated under the identical 0.4 acceleration threshold, 7° steering threshold, and official competition-compatible STOPPED handling?