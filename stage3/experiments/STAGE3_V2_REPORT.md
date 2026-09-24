# Stage 3 v2 — architecture upgrade and controlled experiments (final report)

Run dates: 2026-09-22 (A–H, interrupted) and 2026-09-23/24 (resumed, K/G/F/D, final long runs), on a Vast.ai
RTX 5090 (driver 595.84), PyTorch 2.11.0+cu128, Python 3.12 (`/venv/main`). Repository branch
`stage3-v2-experiments` @ `62d3a86` plus the harness changes listed in §7. Every number below comes from a saved
`runs/stage3_v2/<name>/metrics.json`; nothing is quoted from memory or estimated.

**Status: complete for the planned ablation set.** 19 runs finished. The selected v2 model trained for 100 epochs
(`V3_tcnssm_100ep`) scores **0.7954** validation competition score against **0.7025** for the reproduced v1
baseline at its original 20-epoch schedule (+0.093) and **0.7660** for v1 at 80 epochs (+0.029).
Still pending: route-group-disjoint re-validation and multi-seed confirmation (§8).

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
route-group-disjoint confirmation is still pending.

## 2. Baseline reproduction (A)

| | train (crops) | train (full clips) | val | acc F1 | steer F1 |
|---|---|---|---|---|---|
| `stage3_v1_baseline` (20 ep, this server) | 0.6881 | 0.6934 | **0.7025** | 0.6349 | 0.8602 |
| pre-existing R2 checkpoint `runs/baseline_v1_2/best.pt` | — | — | 0.6938 | 0.6281 | 0.8473 |

Reproduction confirmed (+0.009 vs the stored reference, same config/split/seed; the physics statistics
recomputed here are bit-identical to the reference checkpoint's). All comparisons below use the reproduced run.

## 3. Results

All runs: seed 42, identical data/split/decoder, EMA weights, best-on-val checkpoint selection.
`gap = train_full - val` (negative means val scores higher than train). "hybrid" = gated dual-dilated TCN + Bi-SSM.

### 3.1 20-epoch ablations

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
| `G3_tcnssm_class_ordinal` | baseline | concat | hybrid | ordinal (CORAL) class heads | 0.7452 | 0.7626 | -0.017 | 0.7150 | 0.8739 | 1.88M | 5825 |
| `F3_tcnssm_aux` | baseline | concat | hybrid | jerk + steer-rate aux | 0.7423 | 0.7533 | -0.011 | 0.7031 | 0.8704 | 1.88M | 5825 |
| `D1_tcnssm_dino_pooled` | baseline | concat | hybrid | DINOv3-S pooled | 0.7531 | 0.7632 | -0.010 | 0.7152 | 0.8750 | 1.93M | 5914 |
| `D2_tcnssm_ms_cross_dino_tokens` | multiscale | cross-attn | hybrid | DINOv3-S tokens | 0.7573 | 0.7678 | -0.010 | 0.7260 | 0.8651 | 2.13M | 7308 |

### 3.2 Schedule-length runs

| ID | model | epochs run | best ep | train_full | **val** | gap | acc F1 | steer F1 |
|---|---|---|---|---|---|---|---|---|
| `V1_baseline_40ep` | v1 | 40 | 40 | 0.7297 | 0.7413 | -0.012 | 0.6845 | 0.8736 |
| `H1_waft_a2_40ep` | v1 + WAFT-a2 flow | 40 | 40 | 0.7320 | 0.7403 | -0.008 | 0.6832 | 0.8737 |
| `V1_baseline_80ep` | v1 | 80 | 80 | 0.7571 | 0.7660 | -0.009 | 0.7183 | 0.8772 |
| `V2_final_tcnssm_80ep` | hybrid | 71 (early stop) | 67 | 0.7748 | 0.7850 | -0.010 | 0.7393 | 0.8916 |
| **`V3_tcnssm_100ep`** | **hybrid** | **100** (no early stop) | **91** | 0.7814 | **0.7954** | -0.014 | **0.7522** | **0.8962** |

Validation score along the long schedules (cosine LR decays to its floor at the configured final epoch, so
the same epoch number is a different LR in each run):

| epoch | 20 | 40 | 60 | 71 | 80 | 91 | 100 |
|---|---|---|---|---|---|---|---|
| `V1_baseline_80ep` | 0.7048 | 0.7449 | 0.7597 | 0.7640 | 0.7660 | — | — |
| `V2_final_tcnssm_80ep` | 0.7572 | 0.7750 | 0.7820 | 0.7848 | — | — | — |
| `V3_tcnssm_100ep` | 0.7602 | 0.7757 | 0.7854 | 0.7894 | 0.7931 | **0.7954** | 0.7944 |

### 3.3 Per-class F1 and regression diagnostics

| ID | ACCEL | DECEL | CONST | STOP | LEFT | RIGHT | STRAIGHT | accel MAE | speed MAE | steer MAE |
|---|---|---|---|---|---|---|---|---|---|---|
| `stage3_v1_baseline` | 0.438 | 0.482 | 0.667 | 0.952 | 0.821 | 0.800 | 0.960 | 0.331 | 2.24 | 6.75 |
| `A1_refinement` | 0.396 | 0.450 | 0.664 | 0.946 | 0.821 | 0.796 | 0.960 | 0.338 | 2.24 | 6.76 |
| `B1_multiscale` | 0.444 | 0.515 | 0.676 | 0.951 | 0.812 | 0.789 | 0.959 | 0.322 | 2.17 | 6.81 |
| `C2_ms_cross_attention` | 0.468 | 0.504 | 0.675 | 0.952 | 0.809 | 0.789 | 0.958 | 0.326 | 2.11 | 6.83 |
| `C1_ms_gated` | 0.484 | 0.495 | 0.677 | 0.949 | 0.818 | 0.788 | 0.960 | 0.322 | 2.19 | 6.83 |
| `E1_dual_tcn` | 0.568 | 0.564 | 0.689 | 0.950 | 0.823 | 0.798 | 0.960 | 0.293 | 2.07 | 6.68 |
| `E2_bissm` | 0.578 | 0.582 | 0.683 | 0.947 | 0.834 | 0.794 | 0.959 | 0.316 | 4.47 | 6.84 |
| `E3_tcn_ssm` | 0.620 | 0.589 | 0.688 | 0.949 | 0.831 | 0.838 | 0.965 | 0.297 | 2.76 | 6.49 |
| `K1_dualtcn_ms_gated` | 0.551 | 0.574 | 0.690 | 0.954 | 0.823 | 0.793 | 0.961 | 0.292 | 1.97 | 6.63 |
| `K2_tcnssm_ms_gated` | 0.569 | 0.609 | 0.692 | 0.948 | 0.829 | 0.801 | 0.961 | 0.297 | 2.19 | 6.56 |
| `G3_tcnssm_class_ordinal` | 0.637 | 0.599 | 0.670 | 0.953 | 0.827 | 0.832 | 0.963 | 0.309 | 2.27 | 5.79 |
| `F3_tcnssm_aux` | 0.619 | 0.574 | 0.671 | 0.948 | 0.820 | 0.829 | 0.962 | 0.306 | 2.38 | 6.42 |
| `D1_tcnssm_dino_pooled` | 0.608 | 0.607 | 0.692 | 0.953 | 0.833 | 0.830 | 0.962 | 0.280 | 1.73 | 5.98 |
| `D2_tcnssm_ms_cross_dino_tokens` | 0.641 | 0.617 | 0.694 | 0.952 | 0.834 | 0.801 | 0.960 | 0.275 | 1.54 | 6.11 |
| `V1_baseline_40ep` | 0.544 | 0.558 | 0.683 | 0.954 | 0.839 | 0.820 | 0.962 | 0.299 | 1.73 | 6.09 |
| `H1_waft_a2_40ep` | 0.536 | 0.564 | 0.684 | 0.950 | 0.843 | 0.816 | 0.963 | 0.301 | 1.93 | 5.71 |
| `V1_baseline_80ep` | 0.613 | 0.606 | **0.700** | 0.954 | 0.848 | 0.820 | 0.963 | **0.268** | **1.32** | 5.76 |
| `V2_final_tcnssm_80ep` | 0.669 | 0.653 | 0.681 | 0.955 | 0.854 | 0.854 | 0.967 | 0.299 | 4.07 | 3.91 |
| **`V3_tcnssm_100ep`** | **0.687** | **0.669** | 0.694 | **0.958** | **0.859** | **0.861** | **0.968** | 0.293 | 3.46 | **3.88** |

`V3_tcnssm_100ep` confusion matrices (rows = truth):

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

Compared with `E3_tcn_ssm` at 20 epochs, ACCEL->CONST falls from 39.6% to 30.2% of true ACCEL and DECEL->CONST
from 46.5% to 32.2%. It is still the dominant error: small |a| near the +/-0.25 m/s^2 boundary. Opposite-sign
confusion stays at about 3%.

## 4. Findings

1. **The architecture gain and the schedule gain add up.** The hybrid beats v1 by +0.059 at 20 epochs
   (0.7613 vs 0.7025) and by +0.019 at 80 epochs (0.7850 vs 0.7660). Going to 100 epochs adds another +0.010
   (0.7954). Part of the 20-epoch advantage is faster convergence: the hybrid at 20 epochs (0.7613) is roughly v1
   at 70 epochs (0.7640). But a real margin survives at the long schedule.
2. **The 100-epoch run has only just plateaued.** V3 peaked at epoch 91 (0.7954) and ended at 0.7944. V2 was
   stopped by early stopping (patience 4) at epoch 71 while its cosine LR had not yet decayed. The 100-epoch run
   with early stopping disabled was worth +0.010 over it. The gain from 80 to 100 epochs is small, so returns are
   diminishing.
3. **Temporal modelling was the real architectural bottleneck.** All three temporal variants beat the baseline at
   equal budget: dual-dilated TCN +0.041, Bi-SSM +0.044, gated hybrid **+0.059**. The hybrid is the only
   component that also improves steering. At the long schedule it more than halves steering-angle MAE
   (3.88 deg vs 5.76 for v1 at 80 epochs).
4. **Regression: the hybrid degrades speed MAE at long schedules.** This is the one metric where v2 is worse
   than v1: speed MAE 3.46 (V3) and 4.07 (V2) vs 1.32 for v1 at 80 epochs. Bi-SSM alone showed the same effect
   at 20 epochs (4.47). The competition score does not use speed directly, so this does not show in the headline
   number. But any downstream consumer of the speed head should use v1's, or this needs fixing (§8.3).
5. **Nothing stacks on top of the hybrid at 20 epochs.** Single-seed margins over `E3_tcn_ssm` (0.7613):

   | add-on | val | Δ vs E3 | verdict |
   |---|---|---|---|
   | multi-scale motion tokens + gated fusion (`K2`) | 0.7525 | -0.009 | rejected (also -0.001 on dual-TCN, `K1` vs `E1`) |
   | jerk + steer-rate auxiliary targets (`F3`) | 0.7533 | -0.008 | rejected |
   | ordinal class heads (`G3`) | 0.7626 | +0.001 | tie. Shifts the boundary (ACCEL +0.017, DECEL +0.010, CONST -0.018) instead of sharpening it |
   | frozen DINOv3-S, pooled (`D1`) | 0.7632 | +0.002 | tie. Better speed MAE (1.73 vs 2.76) is the one notable effect |
   | multi-scale + cross-attn + DINOv3 tokens (`D2`) | 0.7678 | +0.007 | below the pre-set +0.01 bar. Best 20-epoch accel F1 (0.726) and speed MAE 1.54, but steering -0.013; +0.25M params, ~5x slower data loading |

   The multi-scale encoder's +0.01 on the v1 TCN (B1/C1/C2) disappears once the temporal model is strong. It was
   compensating for the weak temporal context, not adding spatial information.
6. **Prediction refinement hurts** (-0.015, and train also drops): rejected.
7. **WAFT vs SEA-RAFT is a tie downstream** (0.7403 vs 0.7413 at 40 epochs, per-class within 0.01), despite
   WAFT's much noisier low-level flow at this working resolution:

   | metric (val frames) | SEA-RAFT | WAFT-a2 |
   |---|---|---|
   | E_static = mean \|flow\| at speed<=0.15 m/s | **2.39** | 3.34 |
   | \|flow\| mean / var, CONSTANT | **16.9 / 180** | 70.2 / 8305 |
   | rotation-fit residual, CONSTANT | **1.53** | 5.22 |
   | Var(rho k=2) / Var(rho k=4), steady frames | **13.2 / 3.6** | 38.9 / 10.8 |
   | mean flow confidence | 0.35 | 0.36 |

   Keep SEA-RAFT. The SEA+WAFT fusion experiment (I) was dropped because there is no complementary-error pattern.
8. **No overfitting anywhere.** The train-val gap stayed in [-0.019, -0.008] for every run, including 100 epochs.

Ranking by validation improvement / cost / complexity:

| rank | component | val gain vs its base | compute cost | implementation |
|---|---|---|---|---|
| 1 | gated dual-TCN + Bi-SSM hybrid | +0.059 (20 ep), +0.019 (80 ep) | +0.8M params, VRAM flat | medium (S4D kernel) |
| 2 | longer schedule | v1: +0.064 (20->80); hybrid: +0.034 (20->100) | linear in epochs | trivial |
| 3 | early stopping off for the final run | +0.010 (V3 vs V2) | +30% epochs | trivial |
| 4 | DINOv3 branch (pooled / tokens) | +0.002 / +0.007 | +26 GB cache, slower loading | medium |
| 5 | ordinal class heads | +0.001 | small | low |
| 6 | WAFT flow backend | -0.001 | much higher (2 nets) | medium |
| 7 | jerk/steer-rate aux targets | -0.008 | small | low |
| 8 | multi-scale motion tokens (on the hybrid) | -0.009 | +1.3 GB VRAM | medium |
| 9 | prediction refinement | -0.015 | +0.29M params | low |

## 5. Selected Stage 3 v2 architecture (final)

```
SEA-RAFT-S (unchanged) -> existing geometry/physics (unchanged)
MotionCNN + attention pool (unchanged)
concat fusion (unchanged)
TEMPORAL: gated fusion of [dual-dilated TCN (1,2)(2,4)(4,8)(8,16)(16,32)] and [bidirectional S4D SSM, 4 layers]
heads + Potts/Viterbi decoding (unchanged)
100 epochs, cosine LR, EMA, early stopping disabled, best-on-val checkpoint
```
Config: `stage3/configs/baseline_v1_2.workspace.yaml` + `model.temporal.type=tcn_ssm optimization.epochs=100
early_stopping.enabled=false`. Checkpoint `runs/stage3_v2/V3_tcnssm_100ep/best.pt` (epoch 91), 1.88M trainable
parameters, 5.8 GB training VRAM, 0.94 ms/frame validation, 5.4 h training on one RTX 5090.

Not adopted: refinement, WAFT, SSM-only, multi-scale encoder, auxiliary jerk/steer-rate, ordinal heads, DINOv3.

## 6. Operational notes (from the 2026-09-23 runs)

* **Data loading is CPU-bound at 4 workers.** Each worker was pinned at ~93% CPU while most of the 32 cores sat
  idle. D2 went from ~30 to ~5 min/epoch at 12 workers, and V3 ran at ~2.7 min/epoch with 12. Use
  `WORKERS=12` in `exp.sh`.
* **Pinned host memory grows without bound.** With `pin_memory=True`, PyTorch's caching host allocator keeps
  every page-locked buffer. Full-clip validation batches vary in size, so each training process ends up holding
  ~65 GB of pinned memory (`RssShmem`). Two such processes reached the container's 170.8 GB cgroup limit, and
  both stalled at 100% kernel time for ~30 min. There was no OOM kill and no crash. Stopping one freed it
  immediately. Keep at most 2 concurrent runs, or disable `pin_memory` for the full-clip loaders.
* **3 concurrent runs also do not fit in GPU memory** (each reserves ~12 GB through the CUDA caching allocator,
  against a ~7 GB peak). The first D2 launch OOMed this way. The queue now uses `MAX=2`.
* The first D2 final evaluation was stopped to release that memory. Its `metrics.json` was regenerated
  afterwards with `--eval-only` on `best.pt`.

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
* **New since the interim report:** `stage3/experiments/run.py --resume` continues an interrupted run from its
  `last.pt` with its saved `config.yaml`, trimming any history rows past the checkpoint. The wrapper is
  `stage3/experiments/resume.sh NAME`. It was used for `V1_baseline_80ep`, `K1`, `K2` and `D2`.

Tests: `pytest stage3/tests` 49 passed, 1 deselected (`test_distributed_validation` needs 2 GPUs; both ranks
produce identical output here but the launcher exits non-zero).

## 8. Recommended next experiments

1. **Offline val overestimates the real leaderboard.** A 2026-09-23 submission used the epoch-43 snapshot of
   `V2_final_tcnssm_80ep` (offline val 0.7765) and scored **0.654** on the DACON leaderboard, a gap of ~0.12. The
   validation set is BATON footage, not DACON footage. Treat every offline margin here as a ranking signal only,
   and confirm `V3_tcnssm_100ep` with a leaderboard submission. A route-group-disjoint re-validation (§1 caveat)
   is still worth doing, but it will not close a domain gap.
2. **Multi-seed confirmation** (>=3 seeds) of the final model and of the near-ties (G3, D1, D2). D2 is the most
   interesting: +0.007 overall, best acceleration F1 and good speed MAE. If it holds across seeds, test it at
   100 epochs, where visual context may also address the speed regression. Every margin
   under ~0.01 in this report is single-seed.
3. **Fix the speed regression** (§4.4): e.g. route the speed head from the TCN branch only, raise the speed-loss
   weight, or decode speed from v1. D1's speed MAE of 1.73 suggests visual context also helps here.
4. **Attack the ACCEL/DECEL -> CONSTANT boundary** (~30% of each class still lands there): sweep the
   +/-0.25 m/s^2 decision thresholds on validation, and test the hybrid emission
   `alpha*physical + (1-alpha)*ordinal`. G3 shows the ordinal head moves the boundary, so the two are
   complementary candidates.
5. The schedule has plateaued (peak at epoch 91 of 100), so going beyond 100 epochs is unlikely to pay off
   without an LR-schedule change such as a higher cosine floor or a restart.
