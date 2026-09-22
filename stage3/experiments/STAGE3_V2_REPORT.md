# Stage 3 v2 — architecture upgrade and controlled experiments (interim report)

Run date: 2026-09-22 on a Vast.ai RTX 5090 (driver 595.84), PyTorch 2.11.0+cu128, Python 3.12 (`/venv/main`).
Repository `main` @ `c437a5b` plus uncommitted Stage 3 v2 code (see *Code changes*). Every number below comes
from a saved `runs/stage3_v2/<name>/metrics.json`; nothing is quoted from memory or estimated.

**Status: interim.** 10 runs finished, 3 were interrupted mid-training at the user's request (checkpoints kept),
6 planned ablations were not started. See `stage3/experiments/HANDOFF.md` for the resume plan.

## 1. Baseline architecture (as implemented, verified by inspection)

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
Trainable parameters 1,105,511 (all trainable; SEA-RAFT frozen and outside the trained graph — flow is cached).
Training VRAM 5.81 GB, validation 1.23 ms/frame. Losses (weights): accel_direct 1.00, tied-ordinal 0.50,
stopped 0.50, speed 0.20, accel/speed consistency 0.10, steering 0.25, yaw aux 0.10.

**Dataset/split.** BATON-Sample, 2,159 thirty-second segments (1,693 train / 466 val), route-disjoint by
recording. **Caveat:** routes 24 and 25 have sibling recordings on both sides (`route_24_1` val vs `route_24_2`
train; `route_25_1` val vs `route_25_2/3` train). This is not clip-level leakage, but same-road leakage is
possible. The split was kept unchanged so results stay comparable to the pre-existing R2 reference run; a
route-group-disjoint confirmation is listed as pending work.

## 2. Baseline reproduction (A)

| | train (crops) | train (full clips) | val | acc F1 | steer F1 |
|---|---|---|---|---|---|
| `stage3_v1_baseline` (20 ep, this server) | 0.6881 | 0.6934 | **0.7025** | 0.6349 | 0.8602 |
| pre-existing R2 checkpoint `runs/baseline_v1_2/best.pt` | — | — | 0.6938 | 0.6281 | 0.8473 |

Reproduction confirmed (+0.009 vs the stored reference, same config/split/seed; the physics statistics
recomputed here are bit-identical to the reference checkpoint's). All comparisons below use the reproduced run.

## 3. Results

All runs: seed 42, identical data/split/decoder, EMA weights, best-on-val checkpoint selection.
`gap = train_full - val` (negative means val scores higher than train).

| ID | motion enc | fusion | temporal | extra | epochs | train_full | **val** | gap | acc F1 | steer F1 | params | VRAM MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `stage3_v1_baseline` | baseline | concat | TCN | — | 20 | 0.6934 | 0.7025 | -0.009 | 0.6349 | 0.8602 | 1.11M | 5807 |
| `A1_refinement` | baseline | concat | TCN | 1 refine stage | 20 | 0.6753 | 0.6873 | -0.012 | 0.6137 | 0.8590 | 1.40M | 5811 |
| `B1_multiscale` | multiscale | concat | TCN | — | 20 | 0.6950 | 0.7087 | -0.014 | 0.6467 | 0.8533 | 1.17M | 7149 |
| `C2_ms_cross_attention` | multiscale | cross-attn | TCN | — | 20 | 0.6966 | 0.7104 | -0.014 | 0.6497 | 0.8521 | 1.30M | 7152 |
| `C1_ms_gated` | multiscale | gated FiLM | TCN | — | 20 | 0.7002 | 0.7124 | -0.012 | 0.6510 | 0.8555 | 1.18M | 7148 |
| `E1_dual_tcn` | baseline | concat | dual-dilated TCN | — | 20 | 0.7292 | 0.7430 | -0.014 | 0.6928 | 0.8603 | 1.52M | 5813 |
| `E2_bissm` | baseline | concat | Bi-SSM (S4D) | — | 20 | 0.7346 | 0.7468 | -0.012 | 0.6974 | 0.8621 | 1.09M | 5799 |
| **`E3_tcn_ssm`** | baseline | concat | **dual-TCN + Bi-SSM, gated** | — | 20 | 0.7428 | **0.7613** | -0.019 | **0.7114** | **0.8780** | 1.88M | 5815 |
| `V1_baseline_40ep` | baseline | concat | TCN | — | 40 | 0.7297 | 0.7413 | -0.012 | 0.6845 | 0.8736 | 1.11M | 5807 |
| `H1_waft_a2_40ep` | baseline | concat | TCN | **WAFT-a2 flow** | 40 | 0.7320 | 0.7403 | -0.008 | 0.6832 | 0.8737 | 1.11M | 5799 |

Per-class F1 and regression diagnostics:

| ID | ACCEL | DECEL | CONST | STOP | LEFT | RIGHT | STRAIGHT | accel MAE | speed MAE | steer MAE |
|---|---|---|---|---|---|---|---|---|---|---|
| `stage3_v1_baseline` | 0.438 | 0.482 | 0.667 | 0.952 | 0.821 | 0.800 | 0.960 | 0.331 | 2.24 | 6.75 |
| `A1_refinement` | 0.396 | 0.450 | 0.664 | 0.946 | 0.821 | 0.796 | 0.960 | 0.338 | 2.24 | 6.76 |
| `B1_multiscale` | 0.444 | 0.515 | 0.676 | 0.951 | 0.812 | 0.789 | 0.959 | 0.322 | 2.17 | 6.81 |
| `C2_ms_cross_attention` | 0.468 | 0.504 | 0.675 | 0.952 | 0.809 | 0.789 | 0.958 | 0.326 | 2.11 | 6.83 |
| `C1_ms_gated` | 0.484 | 0.495 | 0.677 | 0.949 | 0.818 | 0.788 | 0.960 | 0.322 | 2.19 | 6.83 |
| `E1_dual_tcn` | 0.568 | 0.564 | 0.689 | 0.950 | 0.823 | 0.798 | 0.960 | **0.293** | 2.07 | 6.68 |
| `E2_bissm` | 0.578 | 0.582 | 0.683 | 0.947 | 0.834 | 0.794 | 0.959 | 0.316 | **4.47** | 6.84 |
| `E3_tcn_ssm` | **0.620** | **0.589** | 0.688 | 0.949 | 0.831 | **0.838** | **0.965** | 0.297 | 2.76 | 6.49 |
| `V1_baseline_40ep` | 0.544 | 0.558 | 0.683 | **0.954** | 0.839 | 0.820 | 0.962 | 0.299 | **1.73** | 6.09 |
| `H1_waft_a2_40ep` | 0.536 | 0.564 | 0.684 | 0.950 | **0.843** | 0.816 | 0.963 | 0.301 | 1.93 | **5.71** |

`E3_tcn_ssm` acceleration confusion (rows = truth):

| | pred ACCEL | pred DECEL | pred CONST | pred STOP |
|---|---|---|---|---|
| ACCEL | 17,389 | 714 | **12,275** | 634 |
| DECEL | 1,295 | 15,101 | **14,469** | 279 |
| CONST | 6,087 | 3,899 | 41,526 | 558 |
| STOP | 344 | 382 | 400 | 23,929 |

The dominant remaining error is ACCEL/DECEL collapsing into CONSTANT (~40% of each), i.e. small |a| near the
+/-0.25 m/s^2 decision boundary — a resolution problem in continuous acceleration, not class confusion between
opposite signs (ACCEL->DECEL is only 4%).

## 4. Findings

1. **The ~0.69 "ceiling" was mostly the 20-epoch schedule, not capacity.** The unchanged v1 architecture scores
   0.7025 at 20 epochs, 0.7413 at 40 epochs, and 0.7640 at epoch 71 of an 80-epoch run (interrupted). Train and
   val rise together throughout and every run's best epoch was its last. Training length is worth about as much
   as the best architecture change.
2. **Temporal modelling was the real architectural bottleneck.** All three temporal variants beat the baseline at
   equal budget: dual-dilated TCN +0.041, Bi-SSM +0.044, and the gated hybrid **+0.059** (0.7613). The hybrid
   also improves steering (0.878 vs 0.860), which no other component did, and reaches in 20 epochs what the
   baseline needs ~70 epochs for.
3. **Bi-SSM alone degrades speed regression** (speed MAE 4.47 vs 2.24 baseline) while improving F1. The gated
   hybrid recovers most of it (2.76): the local TCN branch carries short-horizon dynamics, the SSM branch adds
   global context. This is why the hybrid, not the SSM, is the selected temporal block.
4. **Early spatial pooling is a mild bottleneck, worth ~+0.01.** Multi-scale tokens gain +0.006..+0.010, entirely
   in acceleration (ACCEL/DECEL), while costing ~0.005 steering F1 and +1.3 GB VRAM. Fusion type barely matters
   (gated 0.7124 ~ cross-attn 0.7104 ~ concat 0.7087, spread within noise); gated is the cheapest.
   Evidence that this stacks on a strong temporal model is *not* established (see interrupted `K1`/`K2`).
5. **Prediction refinement hurts** (-0.015, and train also drops): rejected. It is the only component whose train
   score fell, i.e. an optimization problem, not overfitting.
6. **WAFT vs SEA-RAFT is a tie downstream** (0.7403 vs 0.7413, per-class within 0.01), despite WAFT's much noisier
   low-level flow at this working resolution:

   | metric (val frames) | SEA-RAFT | WAFT-a2 |
   |---|---|---|
   | E_static = mean \|flow\| at speed<=0.15 m/s | **2.39** | 3.34 |
   | \|flow\| mean / var, CONSTANT | **16.9 / 180** | 70.2 / 8305 |
   | rotation-fit residual, CONSTANT | **1.53** | 5.22 |
   | Var(rho k=2) / Var(rho k=4), steady frames | **13.2 / 3.6** | 38.9 / 10.8 |
   | mean flow confidence | 0.35 | 0.36 |

   The robust geometry stage (inlier-weighted rotation/FOE fits, robust rho pooling) absorbs the extra noise, and
   the feared CONSTANT/STOPPED degradation does not appear (CONSTANT 0.684 vs 0.683, STOPPED 0.950 vs 0.954).
   WAFT is run far below its 432x960 training resolution, which likely explains the raw-flow noise.
   **Recommendation: keep SEA-RAFT** (equal accuracy, lower cost). Experiment I (SEA+WAFT fusion) is not
   justified — there is no complementary-error pattern to gate on.
7. **No overfitting anywhere.** The train-val gap stayed in [-0.019, -0.008] for every run (val above train,
   because validation uses full clips and EMA weights while training scores augmented crops).

Ranking by validation improvement / cost / complexity:

| rank | component | val gain vs its base | compute cost | implementation |
|---|---|---|---|---|
| 1 | longer schedule (20->40->80 ep) | +0.039 / +0.061 (running) | linear in epochs | trivial |
| 2 | dual-TCN + Bi-SSM gated hybrid | +0.059 | +0.8M params, VRAM flat | medium (S4D kernel) |
| 3 | Bi-SSM alone | +0.044 | -0.02M params | medium |
| 4 | dual-dilated TCN | +0.041 | +0.4M params | low |
| 5 | multi-scale motion tokens | +0.006..+0.010 | +1.3 GB VRAM | medium |
| 6 | fusion type (gated/cross-attn) | ~0 over concat | small | low/medium |
| 7 | WAFT flow backend | -0.001 | much higher (2 nets) | medium |
| 8 | prediction refinement | -0.015 | +0.29M params | low |

## 5. Selected Stage 3 v2 architecture (provisional)

Supported by the evidence so far:

```
SEA-RAFT-S (unchanged) -> existing geometry/physics (unchanged)
MotionCNN + attention pool (unchanged; multi-scale optional, +0.01, unproven on top of the hybrid)
concat fusion (unchanged)
TEMPORAL: gated fusion of [dual-dilated TCN (1,2)(2,4)(4,8)(8,16)(16,32)] and [bidirectional S4D SSM, 4 layers]
heads + Potts/Viterbi decoding (unchanged)
trained for >= 40, ideally 80 epochs
```
Not adopted: refinement stages, WAFT, SSM-only. Undecided pending the interrupted/unrun experiments:
multi-scale encoder, auxiliary jerk/steer-rate, ordinal/CE/focal class heads, DINOv3 visual branch.

## 6. Interrupted and not-run

| run | state | note |
|---|---|---|
| `V1_baseline_80ep` | stopped at epoch 71/80, val 0.7640 | best/last checkpoints kept; longest-schedule control |
| `K1_dualtcn_ms_gated` | stopped at epoch 19/20, val 0.7404 | dual-TCN + multiscale: below `E1` alone (0.7430) so far |
| `K2_tcnssm_ms_gated` | stopped at epoch 11/20, val 0.7177 | hybrid + multiscale; the key "does spatial stack on temporal" test |
| `G3/G1/G2` ordinal, CE, focal class heads | not started | code implemented and unit-tested |
| `F3` jerk + steer-rate aux targets | not started | code implemented; targets already in the dataset |
| `D1/D2` frozen DINOv3-S branch | not started | 26 GB token cache already built for all 2,159 clips |
| final long-schedule combined model | not started | the actual v2 deliverable |
| route-group-disjoint split confirmation | not started | see split caveat in §1 |
| Experiment I (SEA+WAFT fusion) | dropped | §4.6 — no complementary errors |

Because the combined model was never trained at the long schedule, **the v2 gain cannot yet be separated from
the schedule gain**: E3 (+0.059 at 20 ep) and the 80-epoch baseline (+0.061) are individually measured, but their
combination is not.

## 7. Code changes (all behind config flags; v1 defaults byte-compatible)

* `stage3/model/v2_modules.py` (new): `MultiScaleMotionEncoder`, `ConcatFusion`/`GatedFusion`/`CrossAttentionFusion`,
  `DualDilatedTCN`, `S4DKernel`/`BiSSMBlock`/`BiSSM`, `GatedTemporalHybrid`, `RefinementStage`.
* `stage3/model/model.py`: config-selected encoder/fusion/visual/temporal/refinement; v1 parameter names unchanged
  (the pre-existing R2 checkpoint still loads `strict=True`).
* `stage3/model/heads.py`: optional jerk / steer-rate / ordinal (CORAL) or softmax class heads.
* `stage3/trainer/losses.py`: auxiliary + class losses, deep supervision for refinement stages.
* `stage3/trainer/trainer.py`: validation loss, `history.jsonl` per-epoch log, visual-feature passthrough.
* `stage3/trainer/metrics.py`: per-class train F1, steering per-class val F1.
* `stage3/data/targets.py`, `dataset.py`, `augment.py`: jerk/steer-rate targets (+flip sign), optional visual cache.
* `stage3/utils/checkpoint.py`: uncompressed artifacts load with `mmap=True` (crops read only their frames).
* `stage3/flow/waft.py` (new) + `build_flow_estimator`/`validate_config`: `flow.backend: waft`.
* `stage3/scripts/cache_visual.py` (new): frozen DINOv3-S token cache aligned to motion-cache frames.
* `stage3/experiments/{run,summarize,flow_diagnostics}.py`, `exp.sh`, `queue.sh` (new): experiment harness.

Tests: `pytest stage3/tests` 49 passed, 1 deselected (`test_distributed_validation` needs 2 GPUs; both ranks
produce identical output here but the launcher exits non-zero).

## 8. Recommended next experiments

1. Finish `K2` and `V1_baseline_80ep`; then train `E3` at 80 epochs — the actual v2 candidate — and compare
   against `V1_baseline_80ep`, not the 20-epoch baseline.
2. Push the schedule further (120 epochs) or raise the LR/cosine floor: nothing has yet shown where val turns over.
3. Attack the ACCEL/DECEL -> CONSTANT boundary directly (the confusion matrix says this is ~all of the remaining
   headroom): the unrun ordinal/focal class heads and the hybrid emission `alpha*physical + (1-alpha)*ordinal`,
   plus a sweep of the +/-0.25 m/s^2 decision thresholds on validation only.
4. Only then evaluate the DINOv3 branch and jerk/steer-rate aux targets on top of the selected model.
5. Multi-seed confirmation (>=3 seeds) for any component whose margin is under ~0.01 — B1/C1/C2 are all in that band.
6. Re-validate on a route-group-disjoint split before trusting absolute numbers.
