# Stage 3 architecture v1.2

External training videos use PyAV PTS and nearest-frame selection on an exact
10 Hz grid. DACON private videos use a separate every-frame decoder with fixed
`dt=0.1`. The two contracts cannot be selected through a shared timing flag.

Frozen SEA-RAFT-S produces forward flow and mixture-Laplace confidence at the
configured working resolution. A horizontal-FOV prior determines focal length.
Each flow is decomposed into robust 3-DoF rotational flow and derotated
translation. The focus of expansion gives radial expansion. Dense forward
tracks use full image flow to transport earlier source-grid expansion fields.
Finite-frame expansion is converted to an interval-midpoint estimate. The
`k=2` and `k=4` temporal log ratios correct changing depth using a linear-speed
(trapezoidal displacement) approximation to estimate interval-mean `rho = a/v`.

Each frame becomes a `10 x 96 x 168` tensor containing flow velocity x/y, log
magnitude, expansion, `rho(k=2)`, rho validity, flow confidence, the fixed/static
validity mask, and normalized FOE-centered x/y. The 20-D vector order is:

1. rotation rate x/y/z
2. rotation inlier fraction and residual
3. rho estimates at k=2/k=4
4. rho inlier fractions and residual scales
5. log speed/height proxy and quality
6. far/mid/near expansion
7. static-motion magnitude and still fraction
8. high-pass pitch proxy and actual dt

The image branch is the specified four-stage residual motion CNN with masked
attention pooling to 128-D. The physics branch is `20 -> 64 -> 32`. A single
linear `160 -> 128` plus LayerNorm feeds the bidirectional dilated TCN with
dilations `[1,2,4,8,16]` and a 63-frame receptive field. A shared 64-D layer
feeds two direct acceleration scales, positive speed, STOPPED logit, steering
angle, and optional yaw-rate auxiliary heads.

Direct filtered `aEgo` supervises both acceleration heads. Random-threshold tied
ordinal supervision also uses direct acceleration. Derivatives of ground-truth
speed supervise only the 0.10-weight consistency term. Steering-wheel angle is
the lateral target and the only input to LEFT/STRAIGHT/RIGHT decoding. Configured
physical thresholds produce emissions followed by independent Potts Viterbi
decoding for acceleration and steering.

Validation masks missing timestamps and targets explicitly. Steering Macro-F1
excludes ground-truth STOPPED frames; inference still emits steering labels for
every frame. Sequence lengths are propagated through each TCN layer so batching
with padding preserves predictions throughout each valid sequence.

On CUDA, `geometry/cuda.py` batches robust rotation/FOE fitting and keeps expansion,
transport, and rho tensors on-device across both lags. NumPy is the reference path;
canonical resizing and feature pooling share the existing CPU implementation.
CNN inference is chunked by frames while retaining the complete TCN sequence.
CAN validity and gap boundaries propagate through target interpolation/smoothing.
Validation gathers complete samples and trims distributed batch duplicates before
all metrics and checkpoint/early-stop decisions.

The official Stage 3 score is 0.7 times acceleration Macro-F1 plus 0.3 times
steering Macro-F1, with STOPPED frames excluded from steering. Both train and
validation scores are logged at validation intervals. The training score pools
frame confusion counts from the training crops, without another model pass.
Validation competition score selects the best checkpoint and drives early stopping;
acceleration threshold-grid F1 remains a diagnostic metric.
