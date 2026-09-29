# Stage 3 v1 → v3: architecture, validation andn leaderboard (consolidated report)

Compiled 2026-09-28 from the existing records only; no numbers were recomputed or estimated here.
Sources:

| Source | Content used |
|---|---|
| `stage3/ARCHITECTURE.md` | v1(.2) architecture and feature definitions |
| `stage3/experiments/STAGE3_V2_REPORT.md` | v1 reproduction, 19 ablation runs, V2/V3 schedule runs, selected architecture |
| `stage3/experiments/DRIVEDNA_DATA_SCALING_STATUS.md` | V3 baseline details, V3 re-evaluated at the leaderboard decoder |
| `HANDOFF.md` §1, §4, §8 | submissions, their Stage 3 settings, and leaderboard (LB) scores reported by the user |
| `/workspace/outputs/stage3_v12_regression_analysis/REPORT.md` | V3 vs BATON+DriveDNA regression on the LB |
| `submission_tools/*/stage3_decoder_overrides.json` | decoder thresholds shipped in each zip |

**Naming.** "v1" is the original baseline (`baseline_v1_2` config). "V2" is `V2_final_tcnssm_80ep` and "V3" is
`V3_tcnssm_100ep`: the same v2 architecture (TCN+SSM hybrid) trained with different schedules. Submission numbers
(v1, v3, v10, …) are a separate numbering for the full S1+S2+S3 zips.

## 0. Summary

| Stage 3 model | Architecture | Schedule | BATON val (±0.25 / 5°) | LB S3 (best setting) |
|---|---|---|---|---|
| v1 baseline | MotionCNN + physics MLP → bidirectional dilated TCN, 1.11M params | 20 ep | 0.7025 | not submitted on its own |
| v1 baseline | same | 80 ep | 0.7660 | — |
| V2 `V2_final_tcnssm_80ep` | v1 with the temporal model replaced by a gated dual-dilated TCN + Bi-SSM, 1.88M params | 80 ep (early stop at 71, best ep 67) | 0.7850 (ep-43 snapshot: 0.7765) | **0.654** (ep-43 snapshot, submission v1, steer 5°) |
| **V3 `V3_tcnssm_100ep`** | same as V2 | 100 ep, no early stop, best ep 91 | **0.7954** (0.8305 at ±0.5 / 5°) | **0.7475** (submission v10, accel ±0.4, steer 7°) |
| V3 on BATON+DriveDNA (`Baton_DriveDNA_V3`) | same as V3 | 100 ep, best ep 89 | 0.8042 | 0.6572 (submission v12S2, −0.090 vs V3) |

Main points:
* The architecture change (temporal model) and the longer schedule are the whole v1→V3 gain: +0.093 validation
  over v1 at 20 epochs, +0.029 over v1 at 80 epochs.
* Offline BATON validation overestimates the LB by about 0.05–0.12, and after V3 every LB change came from decoder
  thresholds or training data. The weights stayed the same.
* V3 is still the submitted Stage 3 model. The DriveDNA-trained model scored higher offline but lost 0.090 on the LB
  because about half of the DriveDNA training hours are mislabelled (§7.1).

## 1. Data, split and score

* **Data:** BATON-Sample, 41 route folders (`qcamera.mp4` + `vehicle_dynamics.csv`), 2,159 thirty-second segments.
  1,693 train segments from 33 routes (14.02 h), 466 val segments from 8 routes (3.86 h), split by route folder,
  seed 42.
* **Split caveat:** `route_24_1` and `route_25_1` are in val while sibling recordings (`route_24_2`,
  `route_25_2/3`) are in train. This is not clip-level leakage, but the same roads can appear on both sides. The
  split was kept so that results stay comparable. A route-group-disjoint re-validation is still pending.
* **Targets:** CAN is interpolated at actual video times (max gap 0.25 s). Direct `aEgo` is smoothed at 0.5 s and
  1.5 s. STOPPED means `vEgo <= 0.15 m/s`. Steering is the steering-wheel angle in degrees, positive left.
* **Score:** 0.7 × acceleration Macro-F1 + 0.3 × steering Macro-F1. STOPPED frames are excluded from steering. The
  offline decoder uses ±0.25 m/s² and 5°. Val competition score selects the checkpoint.

## 2. v1 baseline architecture

```
RGB 526x330 @20Hz --PTS-> 10 Hz grid --resize--> 192x336
  -> frozen SEA-RAFT-S (5 iters, mixture-Laplace uncertainty -> confidence)
  -> geometry: 3-DoF rotation fit + derotation, FOE, radial expansion, dense forward tracking,
     rho=a/v at lags k=2,4, robust pooling
  -> cache schema 2: motion [T,10,96,168] uint8-quantized + physics [T,20] float32 + raw CAN signals
Spatial branch : MotionCNN 4 stages (32,64,96,128), 1 residual block/stage, masked attention pool -> 128/frame
Physics branch : MLP 20 -> 64 -> 32/frame
Fusion         : concat(160) -> Linear -> LayerNorm -> 128
Temporal       : bidirectional dilated TCN, dilations [1,2,4,8,16], k=3, receptive field 63 frames (6.3 s)
Heads          : shared 128->64 -> {acceleration x2 smoothing scales, softplus speed, STOPPED logit,
                 steering angle, yaw-rate aux}
Decoding       : linear physical margins at +/-0.25 m/s^2 and +/-5 deg -> Potts/Viterbi (penalty 1.0)
```

* **Motion tensor, 10 channels per frame:** flow velocity x/y, log magnitude, expansion, `rho(k=2)`, rho validity,
  flow confidence, static-validity mask, and FOE-centred normalised x/y.
* **Physics vector, 20 dimensions:** rotation rate x/y/z; rotation inlier fraction and residual; rho at k=2 and
  k=4; rho inlier fractions and residual scales; log speed/height proxy and its quality; far/mid/near expansion;
  static-motion magnitude and still fraction; high-pass pitch proxy; actual dt.
* **Size and speed:** 1,105,511 trainable parameters. SEA-RAFT is frozen and its flow is cached, so it is outside
  the trained graph. 5.81 GB training VRAM, 1.23 ms/frame validation.
* **Losses (weights):** accel_direct 1.00, tied-ordinal 0.50, stopped 0.50, speed 0.20, accel/speed consistency
  0.10, steering 0.25, yaw aux 0.10. Full definitions are in §6.
* **Training recipe (shared by all runs):** 96-frame crops, event sampling 0.5, horizontal flip 0.5, batch 8,
  AdamW 3e-4, weight decay 0.05, cosine LR with 5% warmup, bf16, EMA 0.999, seed 42.

**Reproduction:** the v1 baseline retrained on this server scored 0.7025 val (acc F1 0.6349, steer F1 0.8602).
The pre-existing R2 checkpoint `runs/baseline_v1_2/best.pt` scored 0.6938.

## 3. v1 → v2: controlled ablations (20 epochs)

All runs use seed 42 and the same data, split and decoder, with EMA weights and best-on-val checkpoints.
`gap = train_full − val`. "hybrid" = gated dual-dilated TCN + Bi-SSM.

| ID | motion enc | fusion | temporal | extra | train_full | **val** | gap | acc F1 | steer F1 | params | VRAM MB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `stage3_v1_baseline` | baseline | concat | TCN | — | 0.6934 | 0.7025 | -0.009 | 0.6349 | 0.8602 | 1.11M | 5807 |
| `A1_refinement` | baseline | concat | TCN | 1 refine stage | 0.6753 | 0.6873 | -0.012 | 0.6137 | 0.8590 | 1.40M | 5811 |
| `B1_multiscale` | multiscale | concat | TCN | — | 0.6950 | 0.7087 | -0.014 | 0.6467 | 0.8533 | 1.17M | 7149 |
| `C2_ms_cross_attention` | multiscale | cross-attn | TCN | — | 0.6966 | 0.7104 | -0.014 | 0.6497 | 0.8521 | 1.30M | 7152 |
| `C1_ms_gated` | multiscale | gated FiLM | TCN | — | 0.7002 | 0.7124 | -0.012 | 0.6510 | 0.8555 | 1.18M | 7148 |
| `E1_dual_tcn` | baseline | concat | dual-dilated TCN | — | 0.7292 | 0.7430 | -0.014 | 0.6928 | 0.8603 | 1.52M | 5813 |
| `E2_bissm` | baseline | concat | Bi-SSM (S4D) | — | 0.7346 | 0.7468 | -0.012 | 0.6974 | 0.8621 | 1.09M | 5799 |
| **`E3_tcn_ssm`** | baseline | concat | **hybrid** | — | 0.7428 | **0.7613** | -0.018 | 0.7114 | 0.8780 | 1.88M | 5815 |
| `K1_dualtcn_ms_gated` | multiscale | gated | dual-dilated TCN | — | 0.7304 | 0.7422 | -0.012 | 0.6921 | 0.8590 | 1.59M | 7154 |
| `K2_tcnssm_ms_gated` | multiscale | gated | hybrid | — | 0.7421 | 0.7525 | -0.010 | 0.7048 | 0.8636 | 1.95M | 7160 |
| `G3_tcnssm_class_ordinal` | baseline | concat | hybrid | ordinal (CORAL) heads | 0.7452 | 0.7626 | -0.017 | 0.7150 | 0.8739 | 1.88M | 5825 |
| `F3_tcnssm_aux` | baseline | concat | hybrid | jerk + steer-rate aux | 0.7423 | 0.7533 | -0.011 | 0.7031 | 0.8704 | 1.88M | 5825 |
| `D1_tcnssm_dino_pooled` | baseline | concat | hybrid | DINOv3-S pooled | 0.7531 | 0.7632 | -0.010 | 0.7152 | 0.8750 | 1.93M | 5914 |
| `D2_tcnssm_ms_cross_dino_tokens` | multiscale | cross-attn | hybrid | DINOv3-S tokens | 0.7573 | 0.7678 | -0.010 | 0.7260 | 0.8651 | 2.13M | 7308 |

What each change did, measured against its own base model. All margins are single-seed:

| rank | component | val gain | cost | adopted |
|---|---|---|---|---|
| 1 | gated dual-TCN + Bi-SSM hybrid | +0.059 (20 ep), +0.019 (80 ep) | +0.8M params, flat VRAM | **yes** |
| 2 | longer schedule | v1: +0.064 (20→80); hybrid: +0.034 (20→100) | linear in epochs | **yes** |
| 3 | early stopping off for the final run | +0.010 (V3 vs V2) | +30% epochs | **yes** |
| 4 | DINOv3-S branch (pooled / tokens) | +0.002 / +0.007 | +26 GB cache, ~5x slower loading | no (below the +0.01 bar) |
| 5 | ordinal class heads | +0.001 | small | no |
| 6 | WAFT flow backend | −0.001 | two flow networks | no |
| 7 | jerk / steer-rate aux targets | −0.008 | small | no |
| 8 | multi-scale motion tokens (on the hybrid) | −0.009 | +1.3 GB VRAM | no |
| 9 | prediction refinement | −0.015 | +0.29M params | no |

What the ablations show:
* **The temporal model was the real bottleneck.** All three temporal variants beat v1 at equal budget: dual TCN
  +0.041, Bi-SSM +0.044, hybrid +0.059. Only the hybrid also improved steering.
* **The multi-scale encoder stopped helping once the temporal model was strong.** It gave about +0.01 on the v1 TCN
  (B1/C1/C2) but −0.009 on the hybrid (K2). It was making up for weak temporal context rather than adding spatial
  information.
* **WAFT and SEA-RAFT tie downstream** (0.7403 vs 0.7413 at 40 epochs), even though WAFT's flow is much noisier at
  this resolution: static-frame |flow| 3.34 vs 2.39, rotation-fit residual 5.22 vs 1.53. SEA-RAFT was kept.
* **No run overfit.** The train−val gap stayed between −0.019 and −0.008 in every run, including the 100-epoch
  one.

## 4. v1 → V2 → V3: schedule runs

| ID | model | epochs run | best ep | train_full | **val** | gap | acc F1 | steer F1 |
|---|---|---|---|---|---|---|---|---|
| `V1_baseline_40ep` | v1 | 40 | 40 | 0.7297 | 0.7413 | -0.012 | 0.6845 | 0.8736 |
| `H1_waft_a2_40ep` | v1 + WAFT-a2 | 40 | 40 | 0.7320 | 0.7403 | -0.008 | 0.6832 | 0.8737 |
| `V1_baseline_80ep` | v1 | 80 | 80 | 0.7571 | 0.7660 | -0.009 | 0.7183 | 0.8772 |
| `V2_final_tcnssm_80ep` | hybrid | 71 (early stop) | 67 | 0.7748 | 0.7850 | -0.010 | 0.7393 | 0.8916 |
| **`V3_tcnssm_100ep`** | **hybrid** | **100** (no early stop) | **91** | 0.7814 | **0.7954** | -0.014 | **0.7522** | **0.8962** |

Validation score during training. The cosine LR reaches its floor at each run's final epoch, so the same epoch
number means a different LR in each run:

| epoch | 20 | 40 | 60 | 71 | 80 | 91 | 100 |
|---|---|---|---|---|---|---|---|
| `V1_baseline_80ep` | 0.7048 | 0.7449 | 0.7597 | 0.7640 | 0.7660 | — | — |
| `V2_final_tcnssm_80ep` | 0.7572 | 0.7750 | 0.7820 | 0.7848 | — | — | — |
| `V3_tcnssm_100ep` | 0.7602 | 0.7757 | 0.7854 | 0.7894 | 0.7931 | **0.7954** | 0.7944 |

* V2 was stopped by early stopping (patience 4) at epoch 71, before its cosine LR had decayed. Running the full 100
  epochs with early stopping off (V3) added +0.010.
* V3 has plateaued (peak at epoch 91). Training longer is unlikely to help without an LR-schedule change.

### 4.1 Per-class F1 and regression errors

| ID | ACCEL | DECEL | CONST | STOP | LEFT | RIGHT | STRAIGHT | accel MAE | speed MAE | steer MAE |
|---|---|---|---|---|---|---|---|---|---|---|
| `stage3_v1_baseline` (20 ep) | 0.438 | 0.482 | 0.667 | 0.952 | 0.821 | 0.800 | 0.960 | 0.331 | 2.24 | 6.75 |
| `E3_tcn_ssm` (20 ep) | 0.620 | 0.589 | 0.688 | 0.949 | 0.831 | 0.838 | 0.965 | 0.297 | 2.76 | 6.49 |
| `V1_baseline_80ep` | 0.613 | 0.606 | **0.700** | 0.954 | 0.848 | 0.820 | 0.963 | **0.268** | **1.32** | 5.76 |
| `V2_final_tcnssm_80ep` | 0.669 | 0.653 | 0.681 | 0.955 | 0.854 | 0.854 | 0.967 | 0.299 | 4.07 | 3.91 |
| **`V3_tcnssm_100ep`** | **0.687** | **0.669** | 0.694 | **0.958** | **0.859** | **0.861** | **0.968** | 0.293 | 3.46 | **3.88** |

The per-class rows for every ablation run are in `STAGE3_V2_REPORT.md` §3.3.

**Known regression:** the hybrid's speed MAE is worse at long schedules: 3.46 (V3) and 4.07 (V2) vs 1.32 for v1 at
80 epochs. Speed is not part of the competition score, so the headline number doesn't show it. Anything that uses
the speed head should take it from v1 or fix it first.

### 4.2 V3 confusion matrices (BATON val, rows = truth)

| | pred ACCEL | pred DECEL | pred CONST | pred STOP |
|---|---|---|---|---|
| ACCEL | 20,162 | 1,022 | **9,372** | 456 |
| DECEL | 1,011 | 19,944 | **10,033** | 156 |
| CONST | 6,272 | 7,189 | 38,399 | 210 |
| STOP | 250 | 288 | 719 | 23,798 |

| | pred LEFT | pred STRAIGHT | pred RIGHT |
|---|---|---|---|
| LEFT | 9,499 | 1,491 | 164 |
| STRAIGHT | 1,389 | 88,870 | 1,284 |
| RIGHT | 69 | 1,647 | 9,813 |

The main remaining error is ACCEL/DECEL predicted as CONST, which happens for small |a| near the threshold.
From E3 at 20 epochs to V3, ACCEL→CONST fell from 39.6% to 30.2% and DECEL→CONST from 46.5% to 32.2%.
Opposite-sign confusion is about 3%.

## 5. Final v2 architecture (the V3 checkpoint)

```
SEA-RAFT-S (unchanged) -> existing geometry/physics (unchanged)
MotionCNN + attention pool (unchanged)
concat fusion (unchanged)
TEMPORAL: gated fusion of [dual-dilated TCN (1,2)(2,4)(4,8)(8,16)(16,32)] and [bidirectional S4D SSM, 4 layers]
heads + Potts/Viterbi decoding (unchanged)
100 epochs, cosine LR, EMA, early stopping disabled, best-on-val checkpoint
```

* **Config:** `stage3/configs/baseline_v1_2.workspace.yaml` + `model.temporal.type=tcn_ssm`
  `optimization.epochs=100` `early_stopping.enabled=false`.
* **Checkpoint:** `runs/stage3_v2/V3_tcnssm_100ep/best.pt` (epoch 91). R2 copy:
  `r2:car-accident-dataset/stage3/runs/stage3_v2_2026-09-24/runs/`.
* **Size and cost:** 1.88M trainable parameters, 5.8 GB training VRAM, 0.94 ms/frame validation, 5.4 h training on
  one RTX 5090.
* **Offline scores:** val acc F1 0.7522, steer F1 0.8962, score 0.7954; train_full 0.7814. At the LB decoder
  (±0.5 m/s², 5°): acc 0.8023, steer 0.8962, score **0.8305**.
* **Not adopted:** refinement, WAFT, SSM-only, multi-scale encoder, jerk/steer-rate aux, ordinal heads, DINOv3.
* **Code:** new modules are in `stage3/model/v2_modules.py`. All changes sit behind config flags, and the v1 defaults
  are byte-compatible. The v1 checkpoint still loads with `strict=True`.

## 6. Targets, loss objective, training and decoding (details)

This section describes the code as it is now. Unless noted otherwise, v1, V2 and V3 share all of it. The only
V3-specific part is the temporal module (§6.2). Sources: `data/targets.py`, `data/dataset.py`, `data/augment.py`,
`model/heads.py`, `model/v2_modules.py`, `trainer/losses.py`, `trainer/decoder.py`, `inference/predictor.py`,
`submission/model/stage3/runtime.py`, and the saved `runs/stage3_v2/V3_tcnssm_100ep/config.yaml`.

### 6.1 Targets (per 10 Hz frame)

CAN signals are linearly interpolated to the actual video frame times. A value is left missing (NaN) if the
surrounding CAN samples are more than 0.25 s apart. Smoothing uses a Savitzky–Golay filter (polyorder 2) applied
separately to each continuous valid run, with the window set to `round(seconds/dt)` and made odd.

| Target | Definition | Valid when |
|---|---|---|
| `a_long_s1`, `a_long_s2` | direct `aEgo`, smoothed at 0.5 s and 1.5 s | both finite |
| `a_dvdt_s1`, `a_dvdt_s2` | `d/dt` of `vEgo` smoothed at 0.5 s / 1.5 s (`np.gradient`, run edges dropped) | both derivatives valid |
| `speed` | `vEgo` (m/s) | finite |
| `stopped` | `vEgo <= 0.15 m/s` | speed finite |
| `steering_angle` | steering-wheel angle in degrees, positive = left | finite **and not stopped** |
| `yaw_rate_aux` | yaw rate | finite |
| `jerk`, `steer_rate` (v2 aux, not used by V3) | derivatives of the 0.5 s-smoothed accel / steering angle | finite (steer_rate: moving only) |

Direct `aEgo` is required for training, since it is the primary acceleration target. The speed derivative only
feeds the 0.10-weight consistency term.

### 6.2 Heads and temporal module

**Shared head** (`MotionHeads`): `Linear(128→64) → SiLU → Dropout(0.05)`, followed by these linear outputs:

| Output | Shape | Activation | Meaning |
|---|---|---|---|
| `acceleration` | 2 per frame | none | m/s², one per smoothing scale (0.5 s, 1.5 s) |
| `speed` | 1 | softplus | m/s (≥ 0) |
| `stop_logit` | 1 | none (sigmoid → P(stopped)) | STOPPED |
| `steering_angle` | 1 | none | degrees |
| `yaw_rate_aux` | 1 | none | training-only auxiliary |

V3 has no class heads. The optional CORAL ordinal or softmax class heads, and the jerk and steer-rate heads, were
only used in the G3 and F3 ablations.

**Temporal module.**
* **v1:** bidirectional (non-causal) dilated TCN, dim 128, kernel 3, dilations [1,2,4,8,16], replicate padding,
  dropout 0.1, receptive field 63 frames.
* **V2/V3 (`temporal.type=tcn_ssm`):** `GatedTemporalHybrid` computes `out = g·local + (1−g)·global`, where
  `g = σ(MLP([local, global]))` is a per-channel gate (Linear 256→128, SiLU, Linear 128→128).
  * **local = `DualDilatedTCN`:** 5 residual pre-norm blocks with dilation pairs (1,2)(2,4)(4,8)(8,16)(16,32).
    Each block has two same-padded Conv1d streams (kernel 3, SiLU), merged by a sigmoid 1×1-conv gate, then a
    pointwise conv and a residual. Receptive field is 1 + 2·(2+4+8+16+32) = 125 frames (12.5 s).
  * **global = `BiSSM`:** 4 residual pre-norm `BiSSMBlock`s. Each runs a forward and a time-reversed diagonal S4D
    (S4D-Lin, state 32, `dt ∈ [1e-3, 1e-1]`, FFT convolution) with a learned skip term, then GELU, Linear
    256→256, GLU and dropout 0.1. The receptive field is the whole clip in both directions.
  * Padded sequences are masked by length inside both branches.

### 6.3 Loss objective

The total loss is a weighted sum of masked means. Each term is averaged only over frames where the timestamp is
valid and that target's validity mask (§6.1) is true. Where a term uses both acceleration outputs, it is the
average over the two smoothing scales, `½(term(â₁, target₁) + term(â₂, target₂))`.

| Term | Formula | Weight |
|---|---|---|
| `accel_direct` | Huber(δ=1) between `â_k` and `a_long_sk` | **1.00** |
| `ordinal` (tied ordinal) | for each frame, draw 8 random thresholds `τ ~ U(−3, 3)` m/s²; BCE-with-logits of `(â_k − τ)/0.25` against `1[a_long_sk > τ]`, averaged over thresholds | **0.50** |
| `stopped` | BCE-with-logits of `stop_logit` against `stopped` | **0.50** |
| `speed` | Huber(δ=1) between softplus speed and `vEgo` | **0.20** |
| `accel_speed_consistency` | Huber(δ=1) between `â_k` and `a_dvdt_sk` | **0.10** |
| `steering_angle` | Huber(δ=2°) between predicted and true angle, only on non-stopped frames | **0.25** |
| `yaw_aux` | Huber(δ=1) between predicted and true yaw rate | **0.10** |

* **Tied ordinal term:** it supervises the same continuous acceleration output through randomly placed decision
  thresholds rather than through a separate class head. This sharpens the sign and magnitude near class boundaries
  without adding parameters. The temperature of 0.25 matches the ±0.25 m/s² class boundary.
* **Terms not used by V3:** `jerk` (Huber δ=1) and `steer_rate` (Huber δ=10) for F3; CE, focal or cumulative-ordinal
  class losses for G3; and deep-supervision refinement stages (weight 0.5) for A1. Their weights are 0 or absent
  in the V3 config.

### 6.4 Training procedure (V3)

| Item | Setting |
|---|---|
| Samples | 96-frame crops from cached 30 s clips; validation uses full clips |
| Event sampling | with p = 0.5, the crop is centred (±8-frame margin) on a random "event" frame: \|a_long_s1\| > 0.25, \|steer\| > 5°, or a STOPPED transition; otherwise a uniform random crop |
| Augmentation | horizontal flip with p = 0.5: mirror the motion tensor, negate flow-x and FOE-dx channels and physics ω_y/ω_z, negate steering, yaw-rate and steer-rate targets |
| Physics input | 20-D vector standardised with dataset centre/scale (`physics_stats.pt`, stored in the checkpoint); physics dropout 0.05 |
| Optimiser | AdamW, lr 3e-4, weight decay 0.05, gradient clip 1.0, batch 8, no accumulation |
| Schedule | cosine with 5% linear warmup, **100 epochs**, early stopping **off** |
| Precision / averaging | bf16 autocast; EMA of weights with decay 0.999, and the EMA weights are evaluated and shipped |
| Model selection | best validation competition score (§1) at the training decoder (±0.25 / 5°), checked every epoch; V3 picked epoch 91 |
| Seed / data loading | 42 / 12 workers |

### 6.5 Decoding (continuous outputs → labels)

Decoding is pure post-processing and has no trained parameters. The two tasks are decoded independently, each with
a Potts-model Viterbi smoother over the full clip.

**Acceleration (4 classes).**
1. `a = mean(â₁, â₂)` per frame, and `p = σ(stop_logit)`.
2. Emission scores, with `s = emission_scale = 0.5`:
   * ACCEL: `(a − θ₊)/s`
   * DECEL: `(θ₋ − a)/s`
   * CONST: `0`
   * STOP: `logit(p) − logit(0.5)`

   The linear margins cross exactly at the physical thresholds θ₋ and θ₊, so the per-frame argmax is ACCEL above
   θ₊, DECEL below θ₋, CONST in between, and STOP whenever P(stopped) ≥ 0.5 wins. Exact ties are broken away from
   ACCEL/DECEL and toward STOP.
3. **Potts Viterbi:** find the label path that maximises Σₜ score(yₜ) − λ·Σₜ 1[yₜ ≠ yₜ₋₁], with λ =
   `potts_penalty` = 1.0. This removes short flickers: a label switch has to gain more than 1.0 in accumulated
   margin to be worth it. λ = 0 would give the plain framewise argmax.

**Steering (3 classes).** Emission scores, with `s = emission_scale_deg = 5`:
* LEFT: `(angle − θ)/s`
* STRAIGHT: `0`
* RIGHT: `(−angle − θ)/s`

These go through the same Potts Viterbi with λ = 1.0. Steering is decoded for every frame, including STOPPED
frames, but the score ignores STOPPED frames.

**Thresholds.** θ₊/θ₋ and θ are the only decoder values that changed between submissions. `runtime.py` merges
`decoder_overrides.json` into the checkpoint's decoder config at load time, and λ and the emission scales stay at
their training values:

| Setting | accel θ₋ / θ₊ (m/s²) | steer θ (°) | where |
|---|---|---|---|
| training / offline validation | −0.25 / +0.25 | 5 | checkpoint config |
| submission v3 (v4) | −0.25 / +0.25 | 5 (v4: 1.5) | training defaults (v4 changes steering only) |
| NEXAR specialist | −0.5 / +0.5 | 5 | override |
| v5 | −1.0 / +1.0 | 5 | override |
| hybrid | −0.75 / +0.75 | 5 | override |
| v7–v9 | −0.5 / +0.5 | 3 | override |
| **v10–v13, v12S2 (current)** | **−0.4 / +0.4** | **7** | override |

The model was never retrained for these thresholds. Its outputs are regression values, so moving θ only moves the
class boundaries (§7 has the LB effect of each setting).

### 6.6 Inference pipeline (DACON test contract)

1. **Decode.** Every decoded video frame becomes one output row. `dt` is fixed at 0.1 s: the competition states
   Stage 3 input/output is 10 Hz, and PTS is not used for dt at inference. Frames are resized to 192×336.
2. **Flow.** Frozen SEA-RAFT-S produces flow and confidence for each consecutive frame pair.
3. **Features.** Geometry (rotation fit, FOE, expansion, tracking, rho) runs on CUDA and builds the
   `[T,10,96,168]` motion tensor and the `[T,20]` physics vector. Physics is standardised with the checkpoint's
   centre/scale.
4. **Model.** MotionCNN runs in 32-frame chunks. The temporal model, fusion and heads then run over the **whole
   clip at once**, so there is no windowing at test time.
5. **Labels.** The decoder (§6.5) produces the labels, written as `ID, sample_index, accel_label, steer_label`.
   The EMA weights (`ema_model`) are the ones loaded.

## 7. Leaderboard history (Stage 3 score)

The Stage 3 packages in submissions v3, v5, v7, v10, the NEXAR specialist and the length-gated hybrid all use the
**same V3 `best.pt` and code**. They differ only in `stage3_decoder_overrides.json`, so within this group the S3
changes come purely from the decoder thresholds.

| Submission | Stage 3 weights | accel threshold | steer threshold | LB S3 |
|---|---|---|---|---|
| v1 `submit_v1_E3_V2ep43` | V2 ep43 (val 0.7765) | ±0.25 | 5° | 0.654 |
| v3 `submit_v3_P2ens-full349_V3` | V3 | ±0.25 | 5° | 0.7274 |
| v4 `submit_v4_..._steer1p5` | V3 | ±0.25 | 1.5° | 0.6900 |
| v5 `submit_v5_LCv2ens4-motion_V3_acc1_steer5` | V3 | ±1.0 | 5° | 0.6972 |
| length-gated hybrid | V3 | ±0.75 | 5° | 0.7313 |
| NEXAR specialist | V3 | ±0.5 | 5° | 0.7388 |
| v7 `submit_v7_E4E2XN4full_V3_acc0p5_steer3` | V3 | ±0.5 | 3° | 0.7340 |
| **v10 `submit_v10_XCE4-XSE2-XSXN4_V3_acc0p4_steer7`** | V3 | ±0.4 | 7° | **0.7475** (best) |
| v12S2 `submit_v12S2_BatonDriveDNA-S3_acc0p4_steer7` | BATON+DriveDNA V3 recipe (ep 89, val 0.8042) | ±0.4 | 7° | 0.6572 |
| v8, v9, v11, v12, v13 | V3 (v8/v9 ±0.5 / 3°; v11–v13 ±0.4 / 7°) | | | S3 not recorded |

Decoder threshold results on the LB:
* **Acceleration (V3, steer 5°):** ±0.25 0.7274 → **±0.5 0.7388** → ±0.75 0.7313 → ±1.0 0.6972.
* **Steering at accel ±0.5:** 5° 0.7388 vs 3° 0.7340. At ±0.25: 5° 0.7274 vs 1.5° 0.6900. Fitting the threshold on
  the 50 DACON OPEN labels (1.25–1.5°) did not transfer to the LB.
* **Best measured setting:** ±0.4 / 7° (0.7475). No run changes one threshold at a time from ±0.5 / 5°, so how much
  of the gain comes from each threshold is unknown.

**Offline vs LB:** V2 ep43 had val 0.7765 and LB 0.654 (gap about 0.12). V3 had val 0.8305 at ±0.5 / 5° and LB
0.7388 (gap about 0.09). BATON validation footage is not DACON footage, so offline margins are only a ranking
signal. The competition states that Stage 3 input and output are 10 Hz.

### 7.1 Why the BATON+DriveDNA model lost 0.090 (v10 → v12S2)

The two Stage 3 packages are byte-identical except for `best.pt`, so the whole difference comes from the training
data.

| | BATON val @ ±0.4 / 7° | DriveDNA val @ ±0.4 / 7° |
|---|---|---|
| V3 | 0.824 (acc 0.785 / steer 0.915) | 0.782 |
| BATON+DriveDNA | 0.836 (acc 0.805 / steer 0.909) | 0.779 |

* **Acceleration output is compressed.** The predicted-vs-true slope fell from 0.90 to 0.77 on BATON val. On
  DACON OPEN footage the output std is 0.26 vs 0.56 for V3. That acts like raising the threshold, which the LB
  sweep shows costs points.
* **Predictions are about 0.2 s earlier.**
* **Collapse on held-out drives:** Tesla Model 3 drive_180 scored 0.466 (V3 0.760) and RAV4 TSS2 drive_184 scored
  0.395 (V3 0.639), with STOPPED F1 0.00 on drive_184.
* **Root cause:** 21 of 42 DriveDNA training drives are bad (18.6 of 36.5 h, 37% of all training frames). Some have
  placeholder telemetry (KIA EV6, Tesla Model X 204/210/211), some have a video↔CAN offset of at least 1.3 s, and
  one drive's telemetry is unrelated to its video.
* **Fix before any retry:** add signal-validity filters, estimate and correct or drop per-drive offsets, and gate
  the retrained model on per-drive collapse checks and OPEN-footage output amplitude. Keep V3 until then.

## 8. Open items

1. Route-group-disjoint re-validation (the route_24/25 sibling caveat in §1).
2. Multi-seed confirmation (≥3 seeds) of V3 and of the near-ties G3, D1 and D2. Every margin under about 0.01 in
   this report is single-seed.
3. Fix the hybrid's speed regression, e.g. take the speed head from the TCN branch only or raise the speed-loss
   weight.
4. Attack the ACCEL/DECEL→CONST boundary, e.g. a hybrid emission `alpha*physical + (1-alpha)*ordinal`.
5. Retry DriveDNA scaling only after the data fixes in §7.1.
6. Record the S3 scores for v8, v9 and v11–v13 once they are known.
