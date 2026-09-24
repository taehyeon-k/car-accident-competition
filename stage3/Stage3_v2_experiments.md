Stage 3 v2 — Architecture Upgrade and Controlled Experiment Plan

You are working on an existing car-accident driving-dynamics competition repository.

Your task is to inspect the current Stage 3 implementation, reproduce the existing baseline, implement a stronger Stage 3 v2 architecture, and actually run a controlled sequence of experiments.

Do NOT blindly replace the current system.

The current Stage 3 architecture is already a strong and well-generalizing baseline, so all changes must be introduced as modular ablations.

1. Current baseline

First inspect the repository and identify the exact current Stage 3 implementation.

The current model should approximately be:

RGB video
    ↓
Frozen SEA-RAFT-S
    ↓
forward optical flow + uncertainty
    ↓
ego-motion / geometry preprocessing
    ├─ rotational-flow estimation/removal
    ├─ FOE estimation
    ├─ radial expansion
    ├─ dense flow tracking
    ├─ rho ≈ a/v estimates
    └─ physics statistics
    ↓

Spatial branch:
10 × 96 × 168 motion tensor
    ↓
MotionCNN
    ↓
attention pooling
    ↓
128-D/frame

Physics branch:
20-D physics vector
    ↓
MLP
    ↓
32-D/frame

concat
    ↓
160 → 128
    ↓
bidirectional dilated TCN
dilations [1,2,4,8,16]
    ↓
continuous physical heads
    ├─ longitudinal acceleration
    ├─ speed
    ├─ stopped
    └─ steering angle
    ↓
physical thresholds
    ↓
Potts/Viterbi decoding
    ↓
competition labels

Current observed performance with BATON-Sample:

Train F1 ≈ 0.69
Validation F1 ≈ 0.67
Real leaderboard ≈ 0.63

This is important.

The train/validation gap is very small.

Therefore:

do NOT assume overfitting is the main problem;
do NOT increase regularization aggressively;
do NOT replace the existing physics formulation;
treat the current system as a strong low-variance but potentially capacity-limited baseline.

The purpose of Stage 3 v2 is to increase representation capacity while retaining the inductive biases that are already generalizing well.

2. First task: fully inspect the current code

Before editing anything:

Find all Stage 3-related:
models
configs
datasets
BATON adapters
optical-flow preprocessing
geometry extraction
losses
decoders
training loops
inference code
evaluation scripts
Produce a short architecture report containing:
exact tensor shapes;
trainable vs frozen modules;
total/trainable parameter count;
optical-flow backend;
temporal receptive field;
current losses and weights;
current output decoding;
timing/resampling logic.
Do not modify code until the baseline architecture is understood.
3. Reproduce the baseline

Before Stage 3 v2, make sure the current baseline still runs.

Run the existing training/evaluation pipeline using the currently available BATON-Sample data.

Record:

train F1
validation F1
per-class F1
acceleration macro F1
steering macro F1

ACCEL
DECEL
CONSTANT
STOPPED

LEFT
RIGHT
STRAIGHT

Also save:

regression MAE/RMSE for acceleration;
steering-angle MAE;
speed MAE;
confusion matrices;
train/validation losses;
inference runtime;
peak GPU memory.

Store the result as:

stage3_v1_baseline

Do not compare later experiments against remembered numbers only. Compare against the reproduced baseline from the same current code/data split.

4. Stage 3 v2 design philosophy

The existing architecture should NOT be replaced with a generic large video Transformer.

Preserve:

optical motion
+
explicit physics
+
continuous vehicle-state prediction
+
temporal decoding

The proposed improvements are:

preserve spatial information longer;
introduce multi-scale motion features;
optionally add appearance/semantic context;
improve physics-motion fusion;
model both local and long-term temporal behavior;
add prediction refinement;
add useful auxiliary physical targets;
add direct ordinal supervision corresponding to competition states.

The architecture should remain modular so every component can be ablated independently.

5. Stage 3 v2 architecture

Target architecture:

                         RGB
                     /         \
                    /           \
          optical-flow          visual-context
             branch                 branch
                │                      │
                ▼                      ▼
       SEA-RAFT / WAFT          frozen DINOv3-S
                │                      │
                ▼                      ▼
       dense motion tensor       patch features
                │                      │
          multi-scale motion           │
              encoder                  │
                │                      │
          spatial tokens ──────────────┘
                │
                │
20-D physics → physics encoder
                │
                ▼
       physics-conditioned fusion
                │
                ▼
          frame embedding
                │
       ┌────────┴────────┐
       ▼                 ▼
 dual-dilated TCN   bidirectional SSM
 local dynamics      global dynamics
       │                 │
       └────────┬────────┘
                ▼
           gated fusion
                │
         Stage-1 heads
                │
       refinement network
                │
         Stage-2 heads
                │
       continuous predictions
              +
        ordinal predictions
                │
        physics-aware decoding
                │
              F1

Do not implement every component before running experiments.

Implement this incrementally.

6. Experiment A — prediction refinement

This should be the first architecture experiment because it is low-risk.

Current:

temporal features
    ↓
physical heads
    ↓
Viterbi

Change to:

temporal features
    ↓
Stage-1 heads
    ↓
initial:
  acceleration
  speed
  stopped
  steering
    ↓
concatenate predictions with temporal features
    ↓
small temporal refinement network
    ↓
Stage-2 heads
    ↓
Viterbi

Use one refinement stage initially.

Suggested refinement module:

input:
  temporal feature
  stage1 acceleration
  stage1 speed
  stage1 steering
  stage1 stopped probability

→ projection
→ 3–5 lightweight temporal residual blocks
→ final heads

Do not use many refinement stages unless one-stage refinement clearly helps.

Run:

A0 = baseline
A1 = baseline + refinement

Report whether training F1 increases without opening a large validation gap.

7. Experiment B — multi-scale spatial motion encoder

The current architecture appears to collapse the dense flow representation to one vector per frame too early.

Replace or extend the MotionCNN with a multi-scale encoder.

Input remains approximately:

[B,T,10,96,168]

Use a hierarchical CNN:

Stage 1
96×168

Stage 2
48×84

Stage 3
24×42

Stage 4
12×21

Extract motion representations at several scales.

Preserve both:

fine motion
medium motion
global ego motion

Do not globally pool immediately.

Generate spatial tokens.

For example:

fine/global feature maps
      ↓
adaptive pooling
      ↓
4×7 or similar token grid
      ↓
~28 spatial motion tokens/frame

Keep the exact number configurable.

Suggested token dimension:

128–192

Do NOT explode model size.

Run:

B0 = baseline
B1 = multi-scale MotionCNN + existing temporal model
B2 = B1 + refinement

Determine whether early spatial pooling was a bottleneck.

8. Experiment C — physics-conditioned spatial attention

Current fusion approximately:

motion vector + physics vector
        ↓
concatenate

Replace this with explicit conditional fusion.

Encode:

20-D physics
→ MLP
→ physics token

Then use the physics token to attend to the spatial motion tokens.

Possible implementation:

Q = physics/frame token
K,V = spatial motion tokens

cross-attention

or gated FiLM-style modulation if attention is too expensive.

Goal:

The model should be able to emphasize different spatial regions depending on:

rotational-flow quality;
optical-flow confidence;
FOE reliability;
expansion statistics;
speed;
rho stability.

Implement both simple and advanced fusion modes behind config flags:

fusion:
  type: concat | gated | cross_attention

Experiments:

C0 = concat
C1 = gated fusion
C2 = physics-conditioned cross-attention

Compare them fairly.

9. Experiment D — DINOv3-S visual/context branch

Optical flow provides motion but does not identify what is moving.

Add a lightweight frozen DINOv3-S branch.

Important:

DINOv3 must initially remain frozen.

Do NOT fine-tune DINOv3 on BATON-Sample.

Input:

RGB frames

Extract dense patch features.

Reduce them:

DINO feature dimension
→ linear/conv adapter
→ ~64–128 dimensions

Do not globally pool immediately.

Align DINO visual tokens with the motion tokens.

Possible fusion:

motion tokens
+
visual tokens
+
physics token
        ↓
cross-attention

The visual branch should help distinguish:

road surface
vehicle
sky
vegetation
building
road edge
lane region

without explicitly requiring semantic labels.

Frame sampling may be reduced if necessary:

DINO every 2nd or 3rd frame

and interpolate/reuse embeddings temporally.

Experiments:

D0 = motion only
D1 = motion + frozen DINOv3-S pooled feature
D2 = motion tokens + DINO spatial tokens

Monitor GPU memory carefully.

10. Experiment E — stronger temporal architecture

Do NOT immediately replace the TCN.

The current TCN's local temporal inductive bias is useful.

Build a hybrid.

Local branch

Use a dual-dilated TCN.

Instead of a single dilation stream, each block should capture both short and long receptive fields.

Possible pattern:

branch A dilation = d
branch B dilation = larger d

concat/gated merge

Examples:

[1,2]
[2,4]
[4,8]
[8,16]

Keep residual connections.

Global branch

Add a lightweight bidirectional state-space model.

Preferred:

Mamba / Mamba2 / equivalent bidirectional SSM

If implementing bidirectional Mamba cleanly is difficult, use:

forward SSM
+
reversed-sequence SSM

then combine outputs.

Do NOT use an excessively large state dimension.

Suggested hidden dimension:

192 or 256 max initially

Combine:

local TCN output
global SSM output
        ↓
learned gated fusion

For example:

g = sigmoid(MLP([local,global]))

fused =
g * local +
(1-g) * global

Experiments:

E0 = current TCN
E1 = stronger dual-dilated TCN
E2 = Bi-SSM only
E3 = dual-dilated TCN + Bi-SSM

Do not assume E3 must win.

11. Experiment F — auxiliary physical targets

Preserve the existing main targets.

Primary acceleration target:

direct CAN longitudinal acceleration

NOT differentiated speed.

Speed derivative remains auxiliary.

Primary steering target:

CAN steering angle

NOT yaw rate.

Add training-only auxiliary targets.

Jerk
jerk = da/dt

Calculate from appropriately smoothed longitudinal acceleration.

Use robust Huber loss.

Steering velocity
steer_rate = d(steering_angle)/dt

Use robust Huber loss.

These outputs do not need to be used directly at competition inference.

Experiment:

F0 = original continuous heads
F1 = + jerk
F2 = + steering velocity
F3 = + both
12. Experiment G — continuous + ordinal supervision

Keep continuous physical regression.

Add auxiliary discrete/ordinal heads corresponding directly to competition states.

Acceleration:

DECEL
CONSTANT
ACCEL

and retain STOPPED separately.

Steering:

RIGHT
STRAIGHT
LEFT

Do not replace continuous regression.

Train both:

continuous physical objective
+
ordinal/discrete objective

Prefer ordinal structure where appropriate rather than treating states as unrelated categories.

Possible loss:

cross entropy / focal loss

Compare both.

Do NOT let classification become the sole target.

Later experiment with using ordinal logits in Viterbi emissions.

Experiments:

G0 = continuous only
G1 = continuous + CE
G2 = continuous + focal
G3 = continuous + ordinal formulation
13. Experiment H — optical-flow backend

The current baseline uses SEA-RAFT.

Do a tightly controlled comparison against WAFT.

Do NOT change any downstream architecture for the first comparison.

Run:

H0:
SEA-RAFT-S
→ exact existing physics
→ exact existing model

H1:
WAFT
→ exact existing physics
→ exact existing model

Preferred WAFT candidate initially:

WAFT-DAv2-a2

if the pretrained model is available and technically compatible.

Also test DINOv3 WAFT only if practical.

Important concern:

WAFT may be stronger on general benchmarks but may produce noisy low-amplitude flow on nearly static frames.

For this competition that is important because:

CONSTANT
STOPPED

depend heavily on stable low-motion estimates.

Therefore evaluate optical flow not only by final F1.

Log:

mean flow magnitude for STOPPED frames

mean/variance of flow for:
STOPPED
CONSTANT
ACCEL
DECEL

FOE temporal variance

rho_lag2 variance
rho_lag4 variance

rotation fit residual

flow confidence

physics feature variance

Especially compute:

E_static =
mean(||flow||)
for frames with near-zero vehicle speed

and:

Var(rho)
for frames with near-zero longitudinal acceleration

If WAFT:

improves ACCEL/DECEL;
improves LEFT/RIGHT;
but hurts CONSTANT/STOPPED;

do NOT simply discard either model.

Implement optional flow fusion.

14. Optional Experiment I — SEA-RAFT + WAFT fusion

Only do this if H0/H1 show complementary errors.

Compute both:

F_sea
F_waft

Additional features:

F_sea_x
F_sea_y

F_waft_x
F_waft_y

|F_sea - F_waft|

SEA confidence
WAFT confidence

flow magnitudes
physics residuals

Learn a lightweight gate:

g ∈ [0,1]

F_fused =
g * F_waft +
(1-g) * F_sea

The gate may depend on:

SEA uncertainty
WAFT uncertainty
flow magnitude
rotation residual
FOE reliability
rho reliability

Do not implement this unless single-backend experiments justify it because two optical-flow networks significantly increase inference cost.

15. Loss balancing

As auxiliary heads accumulate, manual loss weights may become fragile.

Preserve the current fixed-weight system initially.

Then optionally compare with learned uncertainty weighting.

Do not introduce automatic loss weighting in the same experiment as several architectural changes.

Experiment separately.

Record the learned weights if used.

16. Keep physical decoding

Do NOT remove the physical continuous-to-discrete decoder initially.

Current concept:

continuous acceleration
continuous steering
stopped probability
        ↓
class emissions
        ↓
Potts/Viterbi

Retain it.

Later test a hybrid emission:

E_final =
α * physical_emission
+
(1-α) * ordinal_head_emission

Tune alpha only on validation data.

Do not optimize thresholds directly on leaderboard results.

17. Experiment order

Do NOT implement every idea and run only one final model.

Run controlled experiments in this order.

Phase 1 — cheap/high-information
A. Baseline reproduction
B. Prediction refinement
C. Multi-scale motion encoder
D. Physics-conditioned fusion
Phase 2 — additional information
E. Frozen DINOv3-S branch
F. Auxiliary jerk/steering-rate targets
G. Ordinal heads
Phase 3 — temporal capacity
H. Dual-dilated TCN
I. Bidirectional SSM
J. TCN + SSM fusion
Phase 4 — flow backend
K. SEA-RAFT vs WAFT
L. Optional SEA + WAFT fusion

Only after identifying useful components build the combined Stage 3 v2.

18. Avoid combinatorial search

Do not test every possible combination.

Use greedy controlled selection.

Example:

baseline
   ↓
+ refinement
   ↓
keep only if useful
   ↓
+ multi-scale motion
   ↓
keep only if useful
   ↓
+ better fusion
   ↓
...

A component should normally be retained if:

validation macro F1 improves consistently;
gain is not isolated to one easy class;
train/val gap does not explode;
regression diagnostics do not materially worsen;
inference/memory cost remains acceptable.

Leaderboard results may be recorded when available, but leaderboard score must NOT drive repeated threshold overfitting.

19. Required experiment table

Maintain a machine-readable CSV/JSON plus Markdown table.

Example:

ID	Flow	Motion encoder	Visual branch	Fusion	Temporal	Refinement	Aux heads	Train F1	Val F1	Gap	Acc F1	Steer F1	Params	VRAM
V1	SEA	baseline	none	concat	TCN	no	base	
	
	
	
	
	
	

A1	SEA	baseline	none	concat	TCN	yes	base	
	
	
	
	
	
	

B1	SEA	multiscale	none	concat	TCN	yes	base	
	
	
	
	
	
	

...

Additionally report per-class F1:

ACCEL
DECEL
CONSTANT
STOPPED
LEFT
RIGHT
STRAIGHT
20. Diagnostics

For every serious experiment, produce plots for:

train vs validation F1
train vs validation loss

acceleration:
prediction vs GT over time

steering:
prediction vs GT over time

speed:
prediction vs GT

confusion matrices

class transition errors

For flow experiments additionally plot:

SEA flow magnitude
WAFT flow magnitude

rho estimates
FOE trajectories
rotation estimates

for representative:

STOPPED
CONSTANT
ACCEL
DECEL
LEFT
RIGHT

sequences.

21. Parameter-count discipline

The current model generalizes well.

Do not make Stage 3 v2 unnecessarily huge.

Target approximate trainable parameter budgets:

first target:
<10–20M trainable parameters

acceptable:
<30M

avoid:
large 50M+ temporal heads unless experiments clearly justify them

SEA-RAFT / WAFT and DINOv3 should initially be frozen.

The new capacity should mainly come from:

spatial token encoder
fusion
temporal model
refinement
22. Dataset constraints

Current main experimental dataset is BATON-Sample.

Do not assume BATON-Sample is representative of the final dataset scale.

The architecture should be designed so that it can later scale to:

full BATON
ADAS-TO
possibly additional external driving datasets

Do not over-specialize architecture or normalization specifically to BATON-Sample identities/routes.

Use route/drive-level splitting.

Never split temporally adjacent clips from the same source drive between training and validation if that creates leakage.

Verify the current split implementation explicitly.

23. Timing rules

Keep the current distinction.

External dataset training:

PTS-based decoding
→ resample to 10 Hz

Competition/private inference:

use every competition frame
→ nominal dt = 0.1 s

Do not accidentally apply external-dataset resampling to DACON test inference.

24. Engineering requirements

All new architecture pieces must be configurable.

For example:

model:
  motion_encoder:
    type: baseline | multiscale

  visual_encoder:
    enabled: false
    type: dinov3_vits
    frozen: true

  fusion:
    type: concat | gated | cross_attention

  temporal:
    type: tcn | dual_tcn | ssm | tcn_ssm

  refinement:
    enabled: false
    stages: 1

  auxiliary:
    jerk: false
    steer_rate: false
    ordinal_accel: false
    ordinal_steer: false

flow:
  backend: sea_raft | waft

Do not create multiple duplicated training pipelines.

Use the same dataset/evaluation infrastructure.

25. Reproducibility

Every run must save:

config
git commit hash
random seed
dataset split
checkpoint
metrics.json
predictions
training curves
environment info

Set deterministic seeds where practical.

Experiment directories should have clear names such as:

runs/stage3_v2/
  baseline/
  refinement/
  multiscale/
  physics_attention/
  dinov3/
  dual_tcn/
  tcn_ssm/
  waft/
26. Final combined v2 model

Do NOT define the final model before experiments finish.

Construct it from components that empirically improve validation performance.

The likely candidate is approximately:

RGB
 ├─────────────────────────────┐
 │                             │
 ▼                             ▼
best flow backend         frozen DINOv3-S
 │                             │
 ▼                             ▼
multi-scale motion        visual tokens
tokens                        │
 │                             │
 └─────────────┬───────────────┘
               │
        physics token
               │
               ▼
 physics-conditioned fusion
               │
               ▼
       frame representation
               │
      ┌────────┴────────┐
      ▼                 ▼
dual-dilated TCN      Bi-SSM
      │                 │
      └───────┬─────────┘
              ▼
         gated fusion
              │
        stage-1 heads
              │
      temporal refinement
              │
        stage-2 heads
              │
  continuous + ordinal outputs
              │
     physics-aware Viterbi

But only retain each module if ablations support it.

27. Primary research question

The main question is not:

Can we make the network larger?

The actual question is:

Which additional representations or reasoning mechanisms increase train capacity from the current ~0.69 ceiling while preserving the very small train-validation gap and improving real generalization?

Pay special attention to whether each modification changes:

train F1
validation F1
train-val gap

Interpret patterns.

For example:

train ↑
val ↑ similarly
→ likely useful capacity increase

train ↑↑
val unchanged
→ likely unnecessary capacity / overfit

train unchanged
val unchanged
→ component probably not addressing bottleneck

train and val both ↑
LB later ↑
→ strong candidate
28. Final deliverable

After experiments, create:

stage3/experiments/STAGE3_V2_REPORT.md

The report must contain:

exact original baseline;
architecture diagram;
dataset/split details;
every experiment;
train/val metrics;
per-class F1;
parameter count;
GPU memory;
runtime;
qualitative diagnostics;
WAFT vs SEA-RAFT analysis;
ablation conclusions;
selected Stage 3 v2 architecture;
remaining bottlenecks;
recommended next experiments for full BATON / ADAS-TO.

Also include a concise ranking of experiments by:

validation improvement
compute cost
implementation complexity

Do not claim an improvement unless it is supported by actual experiment output.

29. Important rules
Inspect before modifying.
Reproduce baseline first.
Do not destroy the working v1 implementation.
Keep backward-compatible configs.
Run real experiments, not just implement code.
Do not change several major components in one ablation.
Preserve direct CAN acceleration as the primary acceleration target.
Preserve CAN steering angle as the primary steering target.
Preserve physics-based representations.
Keep optical-flow and DINO backbones frozen initially.
Do not tune using DACON leaderboard repeatedly.
Prefer validation evidence.
Record failures as well as successes.
Stop experiments that clearly diverge or exceed memory unnecessarily.
Commit cleanly after stable milestones.

The objective is to produce a measurably stronger Stage 3 v2, not simply a more complicated architecture.