Your task is to implement the complete Stage 3 acceleration-centric motion TCN pipeline under:

./stage3

The architecture is based on the existing stage3_baseline_architecture_v1_1.md specification, but with exactly FOUR critical corrections described below.

Do NOT redesign the architecture beyond these four corrections unless something is required to make the implementation correct or runnable. Do NOT casually add Transformers, RGB foundation models, detectors, depth networks, new losses, or other architectural components.

The goal is a production-quality, modular, testable Stage 3 implementation that follows the organizational and engineering conventions already used in ./stage2.

⸻

0. FIRST: INSPECT THE EXISTING REPOSITORY

Before writing Stage 3 code, inspect the repository carefully.

Read at minimum:

* stage2/run.py
* stage2/configs/
* stage2/data/
* stage2/model/
* stage2/trainer/
* stage2/utils/
* stage2/scripts/
* stage2/README.md
* stage2/WORKSPACE.md
* root requirements.txt
* root requirements-workspace.txt
* competition/DACON inference-related code if present
* existing checkpoint/logging/W&B utilities
* existing manifest conventions
* existing config-loading conventions
* existing Accelerate setup
* existing checkpoint/resume behavior

Do NOT simply copy Stage 2 code blindly.

Reuse conventions and generic utilities where appropriate, but Stage 3 should be a clean independent Python package.

Before implementation, produce a short implementation plan showing:

1. Stage 2 patterns that will be reused.
2. Proposed Stage 3 directory structure.
3. Dependencies that must be added.
4. Ordered implementation milestones.
5. Any architecture-spec ambiguity that must be resolved.

Then implement the stages one by one.

Do not stop after planning.

⸻

1. CRITICAL CORRECTIONS TO V1.1

These four changes OVERRIDE the corresponding parts of V1.1.

Everything else in V1.1 should remain unchanged unless technically impossible.

⸻

CHANGE 1 — ACCELERATION SUPERVISION

V1.1 incorrectly derives the main acceleration target from smoothed speed.

Change this.

DACON stated that its hidden acceleration categories were generated from BOTH:

* CAN vehicle speed
* CAN longitudinal acceleration

after preprocessing.

Therefore:

Primary acceleration target

Use the source dataset’s synchronized direct longitudinal acceleration signal as the PRIMARY continuous acceleration supervision whenever it is available.

The generic dataset adapter must expose:

signals:
    t             # seconds
    v             # ego vehicle speed, m/s
    a_long        # direct longitudinal acceleration, m/s^2
    steering_angle
    ...

a_long is now a first-class signal.

For BATON / ADAS-TO or any dataset with a suitable synchronized longitudinal acceleration channel, use that signal.

Do not throw it away.

⸻

Direct acceleration preprocessing

Preserve V1.1’s uncertainty over unknown DACON preprocessing by generating TWO filtered versions of the direct longitudinal acceleration:

a_long_s1
a_long_s2

Use the same initial temporal scales as V1.1:

s1 ≈ 0.5 s
s2 ≈ 1.5 s

These are configurable.

Use an appropriate low-pass / Savitzky–Golay-style smoothing pipeline, but DO NOT differentiate these signals.

The primary acceleration heads remain:

μ_a_s1
μ_a_s2

and their main targets are now:

a_long_s1
a_long_s2

not derivatives of speed.

⸻

Speed-derived acceleration becomes AUXILIARY ONLY

Also derive:

a_dvdt_s1
a_dvdt_s2

from the synchronized CAN speed signal.

Compute these from speed using the controlled smoothing/differentiation pipeline.

These are NOT the main labels.

They are an auxiliary consistency signal.

Add a low-weight auxiliary loss:

L_acc_speed_consistency

for example:

Huber(mu_a_s1, a_dvdt_s1)
Huber(mu_a_s2, a_dvdt_s2)

masked wherever derivative quality is poor.

IMPORTANT:

Do NOT differentiate predicted speed.

The model’s acceleration output must remain directly predicted.

This auxiliary consistency compares predicted acceleration against a derivative computed from the GROUND-TRUTH speed signal.

Start with a small loss weight, approximately:

acc_speed_consistency: 0.10

Make it configurable.

The direct CAN acceleration target must dominate.

⸻

Updated acceleration losses

Use approximately:

L_total =
    1.00 * L_acc_direct
  + 0.50 * L_ord
  + 0.50 * L_stop
  + 0.20 * L_speed
  + 0.10 * L_acc_speed_consistency
  + steering-related losses

Do not hard-code these in model code.

Put them in YAML.

⸻

Tied ordinal loss

The tied ordinal loss must now use the filtered DIRECT longitudinal acceleration targets:

a_long_s1
a_long_s2

not speed-derived acceleration.

Preserve V1.1’s random-threshold strategy.

⸻

2. CRITICAL CHANGE — STEERING ANGLE IS THE MAIN LATERAL TARGET

V1.1 predicts yaw rate and derives steering labels from yaw rate.

Change this.

DACON explicitly stated that:

LEFT / STRAIGHT / RIGHT

were generated from synchronized CAN steering angle.

Therefore the primary lateral latent must be steering angle.

⸻

Dataset adapter

Expose:

steering_angle

in a canonical representation.

Prefer the actual wheel/steering angle signal corresponding most closely to the source CAN field.

Record metadata describing:

* units
* sign convention
* whether it is steering-wheel angle or road-wheel angle
* any required conversion

Do not silently mix incompatible angle definitions.

Normalize units internally, preferably degrees or radians consistently across datasets.

Document the choice.

⸻

Model head

Replace the primary yaw-rate output with:

steering_angle_hat

The main lateral regression loss becomes:

L_steering_angle

using Huber loss.

The final:

LEFT / STRAIGHT / RIGHT

classes must be decoded from predicted steering angle with configurable thresholds.

Use the same philosophy as acceleration:

predict the physical continuous latent first, choose the hidden category threshold later.

⸻

Yaw rate

Yaw rate may remain as an AUXILIARY target if it exists or can be derived reliably.

For example:

yaw_rate = curvature * speed

when appropriate.

But yaw rate must NOT determine the final steering category.

If retained, use a low loss weight such as:

yaw_aux: 0.10

and make it optional.

The steering-angle loss should have greater weight.

Example:

steering_angle: 0.25
yaw_aux: 0.10

⸻

Steering decoder

Decode:

RIGHT     if steering_angle_hat < -tau_steer
STRAIGHT  if -tau_steer <= steering_angle_hat <= tau_steer
LEFT      if steering_angle_hat > tau_steer

respecting the verified sign convention of the normalized adapter.

Make tau_steer configurable and tunable on validation data.

If left/right convention differs by source, correct it in the dataset adapter, not inside the model.

STOPPED frames must still produce a steering output for submission, but steering metrics/loss should exclude STOPPED where appropriate.

⸻

3. CRITICAL CHANGE — TRAINING TIMING VS DACON PRIVATE INFERENCE

Do NOT use one frame-resampling implementation for every situation.

There must be TWO explicit timing modes.

⸻

Mode A — external training datasets

For BATON, ADAS-TO, or any other external dataset:

Use:

PTS-based decode
→ determine actual frame timestamps
→ sample onto an exact 10-Hz target grid

Use actual timestamps for feature/kinematic computation.

Implementation requirements:

* decode with PyAV;
* use PTS * time_base;
* validate monotonic timestamps;
* drop duplicate PTS where necessary;
* log large gaps;
* select nearest frame for each 0.1-s target timestamp;
* configurable maximum allowed selection error;
* store actual selected timestamps;
* use actual Δt in geometric formulas;
* generate a validity mask when a grid point cannot be matched.

This timing mode is for TRAINING DATA ADAPTATION.

⸻

Mode B — DACON private inference

The private Stage 3 evaluation videos are already 10 Hz.

DACON explicitly states:

decoded frame index <-> sample_index

is 1:1.

Therefore:

DO NOT resample the private test video.

DO NOT reject decoded frames based on PTS offset.

DO NOT divide frame count by two.

DO NOT try to infer FPS from container metadata.

For competition inference:

frames = decode_every_frame(video)
sample_index = range(len(frames))
dt = 0.1

Every decoded frame produces exactly one Stage 3 row.

The returned DataFrame must contain exactly:

ID
sample_index
accel_label
steer_label

with:

len(output_rows_for_video) == number_of_decoded_frames

Add explicit unit/integration tests for this.

⸻

Timing API

Design the decoder cleanly so timing mode is explicit.

Example:

decode_external_training_video(...)
decode_dacon_stage3_video(...)

or equivalent.

Do NOT bury this distinction in flags scattered throughout the code.

⸻

4. CRITICAL CHANGE — FOCAL MODE DEFAULT

V1.1 currently makes GeoCalib the preferred/default focal estimator.

Change the default to:

calibration:
  focal_mode: prior

Use a configurable horizontal FOV prior:

hfov_prior_deg: ...

and calculate:

f_hat = (W / 2) / tan(HFOV_prior / 2)

GeoCalib should remain supported as an OPTIONAL ablation:

focal_mode: geocalib

but must NOT be required for the default pipeline.

Requirements:

* Stage 3 must run without GeoCalib installed.
* GeoCalib import must be lazy/optional.
* Missing GeoCalib must never break focal_mode: prior.
* focal_mode: known may remain diagnostic only.
* Default configs should use prior.

Do not change the rest of the FOE/derotation/canonical-grid design.

⸻

5. PRESERVE THE REST OF V1.1

Except for the four changes above, implement V1.1 faithfully.

The intended pipeline remains:

external training video
    ↓
PTS decode → exact 10-Hz training grid
DACON private video
    ↓
decode every frame, fixed dt = 0.1 s
    ↓
Frozen SEA-RAFT-S forward optical flow + uncertainty
    ↓
per-video focal prior + FOE + camera-fixed invalid/static mask
    ↓
per-frame 3-DoF rotation estimation
    ↓
derotated flow
    ↓
tracks
    ↓
depth-free rho = a/v motion representation
    ↓
canonical 10-channel motion tensor
+
20-D physics vector
    ↓
Motion CNN  → 128-D
Physics MLP → 32-D
    ↓
concat 160
    ↓
Linear 160 → 128 + LayerNorm
    ↓
Bidirectional dilated TCN
128-D
dilations = [1,2,4,8,16]
RF = 63 frames
    ↓
shared 128 → 64
    ↓
continuous heads:
    acceleration at two scales
    speed
    STOPPED
    steering angle
    optional yaw auxiliary
    ↓
threshold-based emissions
    ↓
Potts Viterbi
    ↓
accel_label + steer_label

⸻

6. DEFAULT MODEL ARCHITECTURE

Implement the architecture cleanly in modular classes.

Suggested components:

stage3/model/
    motion_cnn.py
    physics_mlp.py
    tcn.py
    heads.py
    model.py

Names may differ if Stage 2 conventions suggest better names.

⸻

Motion CNN

Input:

10 × 96 × 168

Default:

Stem:
Conv3x3 stride2 → 32
Stage1:
1 residual block, 32
Down:
32 → 64
Stage2:
1 residual block, 64
Down:
64 → 96
Stage3:
1 residual block, 96
Down:
96 → 128
Stage4:
1 residual block, 128
masked attention pooling
output:
128-D

Residual block:

Conv3x3
GroupNorm(8)
SiLU
Conv3x3
GroupNorm(8)
Residual
SiLU

⸻

Physics MLP

Input:

20-D

Architecture:

20 → 64
LayerNorm
SiLU
Dropout(0.05)
64 → 32

⸻

Fusion

128 + 32 = 160
Linear(160,128)
LayerNorm

No nonlinear fusion network.

⸻

TCN

dim: 128
kernel_size: 3
dilations: [1,2,4,8,16]
dropout: 0.10
causal: false
padding: replicate

Each block:

LayerNorm
Conv1D 128→128, k3, dilation d
SiLU
Dropout
Conv1D 128→128, k1
Dropout
Residual add

Only ONE temporal model.

No Transformer.
No LSTM.
No GRU.

⸻

7. MOTION FEATURES

Implement V1.1’s geometry/motion path.

Important modules should be separated and individually testable.

Suggested structure:

stage3/geometry/
    calibration.py
    foe.py
    rotation.py
    tracks.py
    rho.py
    ground_plane.py
    canonical_grid.py
    physics_features.py

⸻

Optical flow

Frozen SEA-RAFT-S.

Forward flow only by default.

Do not train SEA-RAFT.

Support offline feature caching.

⸻

Core channels

Canonical tensor remains:

1. derotated flow x / dt
2. derotated flow y / dt
3. log flow magnitude
4. eta expansion rate
5. rho map, k=2
6. rho validity
7. flow confidence
8. camera-fixed-invalid/static validity mask
9. FOE-centered dx
10. FOE-centered dy

Do not add RGB.

⸻

rho = a/v

Implement the V1.1 derivation carefully and test it independently.

For baselines:

k = 2
k = 4

Use:

rho_map: k=2
pooled physics:
    rho_hat_2
    rho_hat_4

Preserve quality/inlier features.

Do not substitute raw second-difference optical acceleration as the main representation.

Raw optical acceleration should only exist as an ablation.

⸻

8. PHYSICS VECTOR

Preserve the 20-D V1.1 vector:

1–3   rotation omega_x/y/z
4     rotation inlier fraction
5     rotation residual
6     rho_hat_2
7     rho_hat_4
8–9   rho inlier fractions
10–11 rho residual scales
12    log(v/h + eps)
13    v/h quality
14–16 eta far/mid/near
17    static-motion magnitude
18    still fraction
19    high-passed pitch proxy
20    dt

Normalize using training-split robust statistics only.

Store normalization statistics with the model/checkpoint/config.

⸻

9. UPDATED TARGET GENERATION

Create a dedicated label/target module.

Suggested:

stage3/data/targets.py

It should expose a pure function such as:

make_targets(signals, frame_times, cfg)

and return at least:

{
    "a_long_s1": ...,
    "a_long_s2": ...,
    "a_dvdt_s1": ...,
    "a_dvdt_s2": ...,
    "speed": ...,
    "steering_angle": ...,
    "yaw_rate_aux": ...,
    "stopped": ...,
    "valid_accel": ...,
    "valid_steer": ...,
}

Direct longitudinal acceleration is the primary signal.

Speed derivative is auxiliary.

Steering angle is the primary lateral signal.

⸻

10. UPDATED HEADS

Shared representation:

h_t: 128
→ Linear(128,64)
→ SiLU
→ Dropout(0.05)

Heads:

acceleration:
    Linear(64,2)
    outputs μ_a_s1, μ_a_s2
speed:
    Linear(64,1)
    softplus or appropriate positive transform
stop:
    Linear(64,1)
steering_angle:
    Linear(64,1)
yaw_rate_aux:
    optional Linear(64,1)

Do not create a direct 4-class acceleration classifier unless needed only as an explicit ablation.

Do not create a direct steering 3-class head by default.

⸻

11. UPDATED LOSSES

Use configurable YAML weights.

Default starting point:

loss_weights:
  accel_direct: 1.00
  ordinal: 0.50
  stopped: 0.50
  speed: 0.20
  accel_speed_consistency: 0.10
  steering_angle: 0.25
  yaw_aux: 0.10

yaw_aux should automatically become zero/disabled if unavailable.

Loss definitions:

L_acc_direct:
    Huber(mu_a_s1, a_long_s1)
    Huber(mu_a_s2, a_long_s2)
L_acc_speed_consistency:
    Huber(mu_a_s1, a_dvdt_s1)
    Huber(mu_a_s2, a_dvdt_s2)
L_ord:
    V1.1 random-threshold tied ordinal loss,
    TARGETING DIRECT acceleration a_long_s*
L_stop:
    BCE
L_speed:
    Huber
L_steering_angle:
    Huber(predicted steering angle, CAN steering angle)
L_yaw_aux:
    Huber if available

Mask losses correctly.

Do not allow missing sensor targets to create NaNs.

⸻

12. DECODER

Preserve V1.1 acceleration decoding:

μ_a
+
STOPPED probability
+
configurable acceleration thresholds
+
Potts Viterbi

Steering decoding changes to use:

steering_angle_hat

instead of yaw rate.

Thresholds must be config-driven.

Keep all hidden competition thresholds tunable.

Do not bake them into source code.

⸻

13. DATASET ADAPTERS

Build a generic adapter interface.

Suggested:

stage3/data/adapters/
    base.py
    baton.py
    adas_to.py
    dacon.py

Do not let model/trainer code know dataset-specific CSV names.

Canonical record:

ClipRecord(
    video_path,
    signals,
    group_keys,
    metadata,
)

Canonical signal names:

t
v
a_long
steering_angle
yaw_rate        optional
curvature       optional
valid

Each adapter is responsible for:

* source column names;
* units;
* time base;
* sign convention;
* synchronization;
* missing signals.

⸻

14. STAGE 3 DIRECTORY STRUCTURE

Follow Stage 2’s conventions where reasonable.

A target structure might be:

stage3/
├── __init__.py
├── README.md
├── ARCHITECTURE.md
├── WORKSPACE.md
├── run.py
│
├── configs/
│   ├── baseline_v1_2.yaml
│   ├── baseline_v1_2.workspace.yaml
│   └── pretrained-assets.json
│
├── data/
│   ├── dataset.py
│   ├── sampler.py
│   ├── targets.py
│   ├── timing.py
│   ├── manifest.py
│   ├── cache.py
│   └── adapters/
│       ├── base.py
│       ├── baton.py
│       ├── adas_to.py
│       └── dacon.py
│
├── flow/
│   ├── sea_raft.py
│   └── preprocessing.py
│
├── geometry/
│   ├── calibration.py
│   ├── foe.py
│   ├── rotation.py
│   ├── tracks.py
│   ├── rho.py
│   ├── canonical_grid.py
│   └── physics_features.py
│
├── model/
│   ├── motion_cnn.py
│   ├── physics_mlp.py
│   ├── tcn.py
│   ├── heads.py
│   └── model.py
│
├── trainer/
│   ├── trainer.py
│   ├── losses.py
│   ├── metrics.py
│   └── decoder.py
│
├── inference/
│   ├── predictor.py
│   └── dacon.py
│
├── scripts/
│   ├── build_manifest.py
│   ├── inspect_dataset.py
│   ├── cache_motion.py
│   ├── compute_statistics.py
│   ├── validate_cache.py
│   ├── train.py or reuse run.py
│   ├── evaluate.py
│   └── benchmark_inference.py
│
└── tests/
    ├── test_timing.py
    ├── test_targets.py
    ├── test_geometry.py
    ├── test_rho.py
    ├── test_flip.py
    ├── test_tcn.py
    ├── test_decoder.py
    ├── test_cache.py
    └── test_dacon_inference.py

Do not force this exact tree if Stage 2’s current conventions suggest a cleaner equivalent.

⸻

15. TRAINING INFRASTRUCTURE

Mirror Stage 2’s established infrastructure where appropriate:

* Accelerate
* bf16
* deterministic seeding
* YAML configs
* resume support
* W&B/tracker integration
* checkpointing
* best-checkpoint tracking
* EMA
* gradient clipping
* cosine scheduler
* warmup
* structured output directories

Stage 3 run.py should support:

python -m stage3.run \
  --config stage3/configs/baseline_v1_2.workspace.yaml

and ideally the direct equivalent:

python stage3/run.py \
  --config stage3/configs/baseline_v1_2.workspace.yaml

if Stage 2 supports both.

⸻

16. TRAINING DEFAULTS

Use V1.1 defaults unless overridden:

sampling:
  crop_frames: 96
  event_fraction: 0.5
  event_position_margin: 8
motion_cnn:
  widths: [32,64,96,128]
  blocks_per_stage: 1
tcn:
  dim: 128
  dilations: [1,2,4,8,16]
  dropout: 0.10
  causal: false
  padding_mode: replicate
training:
  precision: bf16
  optimizer: adamw
  learning_rate: 3e-4
  weight_decay: 0.05
  grad_clip_norm: 1.0
  scheduler: cosine
  warmup_ratio: 0.05
  ema_decay: 0.999

⸻

17. VALIDATION

Implement the validation described in V1.1.

At minimum log:

acceleration Macro-F1
steering Macro-F1
per acceleration class F1
STOPPED / CONSTANT confusion
continuous acceleration MAE
direct acceleration MAE
speed MAE
steering-angle MAE
optional yaw MAE
boundary F1 ±0.5 s
boundary F1 ±1.0 s
boundary delay
metrics by speed bin

Primary model-selection metric:

validation acceleration Macro-F1 threshold-grid mean

because acceleration is the critical task.

Use route/driver/vehicle-disjoint validation where metadata permits.

⸻

18. FLOOR BASELINES AND ABLATIONS

Implement enough modularity that these can be run without code rewrites:

F0 majority
F1 physics-only formula
F2 HistGradientBoosting on physics vector
A0 plain derotated flow CNN+TCN
A1a raw optical acceleration
A1b eta + rho
A2 full physics vector
A3 loss ablations
A4 capacity
A5 geometry normalization off
A6 Viterbi off/on
A7 GAP vs attention pooling
A8 degraded flow

Do not implement all experiment runs immediately if it delays core code, but architecture/config switches must allow them.

⸻

19. UNIT TESTS — REQUIRED

Do not consider the implementation complete without tests.

Required tests:

Timing

External synthetic 20-Hz/VFR video:

PTS → correct 10-Hz resampling

DACON mode:

N decoded frames → exactly N sample_index rows

⸻

Direct acceleration target

Synthetic signals:

v(t)
a_long(t)

Verify:

* direct acceleration target comes from a_long;
* derivative-of-speed exists separately;
* changing speed smoothing does NOT silently replace a_long;
* missing a_long is handled explicitly, not silently.

⸻

Steering target

Verify that final LEFT/STRAIGHT/RIGHT classes derive from:

predicted steering angle

not yaw rate.

⸻

Calibration default

Without GeoCalib installed:

focal_mode: prior

must run successfully.

⸻

Synthetic geometry

Test:

* FOE recovery;
* rotation recovery;
* rho_hat ≈ a/v;
* focal scaling invariance;
* correct signs.

⸻

Flip

Double horizontal flip = identity.

Acceleration unchanged.

Steering angle sign reverses.

LEFT ↔ RIGHT.

⸻

TCN

Batch-padded and whole-sequence inference should agree on the valid region.

⸻

Decoder

lambda=0 must reproduce framewise decoding.

⸻

20. CACHING

Implement a versioned motion-feature cache.

Cache should contain the V1.1 canonical motion features and physics features, but NOT hard-coded labels.

Raw synchronized signals must remain available so target-generation settings can change without re-running optical flow.

Cache key must include:

SEA-RAFT model/version
working resolution
focal mode/prior
FOE implementation version
rho implementation version
canonical grid definition
feature-code version

A target-preprocessing change must NOT require regenerating optical flow.

⸻

21. INFERENCE INTEGRATION

Build a clean callable API such as:

predict_stage3(data_dir, model_dir) -> pd.DataFrame

The Stage 3 package should be usable from the competition’s root inference.py.

Do not break Stage 1 or Stage 2.

If root inference.py already exists, integrate Stage 3 in the least invasive way possible.

Private DACON Stage 3 inference contract:

input:
    one or more 10-Hz evaluation videos
output:
    DataFrame columns exactly:
        ID
        sample_index
        accel_label
        steer_label

For each video:

number of output rows == number of decoded frames
sample_index == 0..N-1

No PTS resampling here.

⸻

22. PERFORMANCE

Build:

python -m stage3.scripts.benchmark_inference ...

It must report timing for:

decode
SEA-RAFT
calibration/FOE
rotation
tracks/rho
feature construction
CNN
TCN
decoder
total

The entire competition executes Stage 1 → Stage 2 → Stage 3 under a shared time limit, so Stage 3 runtime must be measurable before final deployment.

Use batching wherever practical.

⸻

23. IMPLEMENTATION ORDER

Do NOT create the whole codebase in one uncontrolled pass.

Implement in this exact order and verify each stage before moving on.

Milestone 1 — repository integration

* inspect Stage 2
* create Stage 3 skeleton
* config loader
* logging/tracking integration
* smoke imports
* basic tests

Run tests.

⸻

Milestone 2 — dataset contract + timing

* base adapter
* BATON adapter
* ADAS-TO adapter
* external PTS→10Hz mode
* DACON every-frame mode
* raw signal representation

Run timing tests.

⸻

Milestone 3 — target generation

Implement:

direct a_long primary
smoothed a_long_s1/s2
speed-derived a_dvdt_s1/s2 auxiliary
speed
steering angle
optional yaw
stop
masks

Run synthetic target tests.

⸻

Milestone 4 — SEA-RAFT wrapper

* frozen model
* batched inference
* uncertainty
* prior focal mode
* cache API

Run smoke test on a short video.

⸻

Milestone 5 — geometry

Implement in isolation:

* FOE
* rotation
* derotation
* tracks
* rho
* v/h
* eta
* physics features
* canonical grid

Build synthetic-scene tests before using real data.

Do not continue if rho fails the synthetic a/v test.

⸻

Milestone 6 — cache pipeline

Implement:

python -m stage3.scripts.cache_motion \
  --config ... \
  --manifest ...

Validate shapes, NaNs, value ranges, storage size and cache hashes.

⸻

Milestone 7 — model

Implement:

* motion CNN
* attention pooling
* physics MLP
* fusion
* TCN
* heads

Write shape/forward/backward tests.

⸻

Milestone 8 — losses + metrics + decoder

Implement corrected losses.

Implement steering-angle decoder.

Implement acceleration decoder.

Implement Potts Viterbi.

Write unit tests.

⸻

Milestone 9 — trainer

Mirror Stage 2’s training conventions.

Implement:

* Accelerate
* bf16
* gradient accumulation
* checkpoint/resume
* EMA
* W&B
* validation
* best checkpoint
* reproducibility

Run a tiny overfit test on 1–2 clips.

The model should be able to intentionally overfit a tiny dataset before a full training run.

⸻

Milestone 10 — full validation

Run the smallest useful A0/F2 baselines first.

Then A1b/A2.

Do not optimize larger capacity until the core geometry is shown to work.

⸻

Milestone 11 — DACON inference

Implement full-sequence Stage 3 inference.

Verify:

decoded frame count == output row count

for every test video.

Benchmark runtime.

⸻

24. CODE QUALITY RULES

* Type hints for public APIs.
* Docstrings for mathematical/geometry functions.
* No giant single files.
* No copy-pasted config constants.
* No dataset-specific paths inside model code.
* No silent fallback when required training labels are missing.
* No hidden internet downloads during competition inference.
* Pretrained assets must have explicit local paths/configuration.
* Clear exceptions with actionable messages.
* Deterministic seeds.
* Avoid unnecessary Python loops over pixels.
* Geometry should use vectorized NumPy/PyTorch operations.
* GPU operations should remain on GPU where worthwhile.
* Do not convert GPU tensors to CPU repeatedly inside per-frame loops.
* Validate finite values after geometry stages.
* Log geometry-quality distributions.

⸻

25. DOCUMENTATION TO PRODUCE

After implementation create/update:

stage3/README.md
stage3/ARCHITECTURE.md
stage3/WORKSPACE.md
stage3/IMPLEMENTATION_AUDIT.md

README.md should contain exact copy-paste commands for:

1. preparing manifests;
2. caching motion;
3. computing statistics;
4. running tests;
5. training;
6. validating;
7. benchmarking inference;
8. running DACON Stage 3 inference.

IMPLEMENTATION_AUDIT.md should map every important V1.2 architecture requirement to the actual source file/function implementing it.

⸻

26. FINAL VERIFICATION BEFORE YOU FINISH

Before reporting completion:

1. Run all Stage 3 unit tests.
2. Run Stage 2 tests/smoke checks to ensure Stage 2 was not broken.
3. Run python -m stage3.run --help.
4. Run cache pipeline on a tiny sample.
5. Run model forward/backward.
6. Run tiny-overfit training.
7. Run Stage 3 inference on at least one sample video.
8. Confirm output schema exactly.
9. Confirm one decoded DACON frame produces one sample_index.
10. Verify no GeoCalib dependency is required by the default config.
11. Verify direct longitudinal acceleration is the primary target.
12. Verify speed-derived acceleration is only auxiliary.
13. Verify steering class decoding uses steering angle, not yaw rate.
14. Verify no architecture components beyond V1.1 + these four corrections were added.
15. Report runtime/memory measurements for the smoke run.

At the end, provide:

* files created;
* files modified;
* commands to reproduce;
* tests executed and results;
* known TODOs;
* unresolved dataset-column mappings;
* runtime numbers;
* exact training command;
* exact cache command;
* exact evaluation command;
* exact DACON inference command.

Do not claim completion if tests are failing.

The priority is correctness and maintainability, not writing the most code possible.