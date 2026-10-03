# Retained architectures

## Stage 1 — global_g1

The retained model is a ConvNeXt-Tiny encoder with local-patch attention, global spatial pooling, temporal convolution blocks, and a burst-profile encoder. Sparse frames provide spatial artifacts; native-rate bursts provide temporal re-recording cues. Gated late fusion weights the sparse, global and burst branches. The preserved checkpoint has threshold 0.5 and no independent residual-profile branch enabled.

The model loads with pretrained initialization disabled, then restores all tensors from the checkpoint. The runtime and checkpoint are unchanged from the recorded 0.953197975 lineage.

## Stage 2 — v8

| Component | Retained implementation |
|---|---|
| Backbone | Geometry-adapted DINOv3-S/16, `phase1_partial_noanchor`, frozen at inference |
| Image processing | Letterbox to 448×800; ImageNet normalization; 7×10 pooled patch tokens |
| Sampling | All frames through 128; 128 samples through 256; 192 through 512; 256 through 768; otherwise 320, selected from normalized 192/383 grids |
| Appearance/temporal head | AuxPyramid / LCPyramid modules |
| Global motion | Phase-correlation camera shift and aligned differences; 14 features per retained frame |
| Residual motion | Lucas–Kanade/RANSAC similarity alignment, then residual spatial statistics; 36 features |
| Motion input | Max and mean between samples: 28 global + 72 residual = 100 dimensions |
| Members | `E4_sa_full`, `E2_sa_full`, `XN4_sa_full`; seeds 0–3 in each family |
| Training distinction | Temporal-rate augmentation at native/half/third rates; E2 adds boundary supervision; XN4 adds weakly supervised NEXAR examples |
| Decoding | Equal probability averages; constrained ENTRY ≤ COLLISION; camera-shift fusion β=0.5; robust-z cap 6; native-frame collision snap |

The runtime reads original frame indices rather than assuming a video FPS. Motion is recomputed over the supplied frame sequence. It retains the training extractor's sequential handling of phase correlation, including OpenCV's in-place windowing behavior. Removing that detail would alter the feature distribution.

The saved v8 refit configs use `phase_rep=none` and `w_phase=w_tr=w_mono=0`. The training objective is the NT direct loss (Gaussian event targets with sigma=1 sampled position, side and evasion weights .5), with boundary loss of weight 1.0 for E2. Phase-study helper files remain only because the original trainer imports their direct-loss and batching functions. No detector, V-JEPA branch, depth model or object-tracking cache is required by v8 inference.

## Stage 3 — V3_tcnssm_100ep

| Component | Retained implementation |
|---|---|
| Flow | SEA-RAFT-S, official source revision identified in checkpoint as `9137517`; frozen; 5 refinement iterations |
| Working image size | 192×336 |
| Motion tensor | 10 channels on a 96×168 canonical grid |
| Physics vector | 20 dimensions, normalized by checkpoint-stored center/scale |
| Motion encoder | Four CNN stages of widths 32, 64, 96, 128; attention pooling |
| Physics encoder | MLP; concatenated with motion representation |
| Temporal model | 128-dimensional gated dual-dilated TCN + bidirectional diagonal state-space model |
| Heads | Acceleration at two smoothing scales, positive speed, stopped logit, steering angle, auxiliary yaw rate |
| Training | BATON; AdamW 3e-4; weight decay .05; cosine schedule; bf16; EMA .999; 100 epochs; selected epoch 91 |
| Best decoder | Acceleration ±0.4 m/s²; steering ±7°; stopped probability .5; Potts penalty 1.0 |

Training losses combine acceleration regression, tied ordinal supervision, stopped BCE, speed regression, acceleration/speed consistency, steering regression and yaw supervision. Their checkpoint weights are respectively 1.0, .5, .5, .2, .1, .25 and .1.

External training videos were decoded using PTS and resampled to 10 Hz. Competition inference processes every supplied video frame and uses fixed 0.1-second spacing, matching the competition contract. STOPPED training labels use CAN speed ≤0.15 m/s; inference uses the learned stopped probability, not a hard speed-head cutoff. Stopped frames are excluded from steering evaluation.

The original BATON validation split may share road segments across related route recordings. Validation scores therefore should not be read as independent estimates of competition performance. The retained method is selected by its recorded leaderboard score.
