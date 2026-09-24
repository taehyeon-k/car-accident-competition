# Stage 2 temporal-pyramid and frame-count experiments

Frozen geometry-DINOv3-S, 349 videos (279 train / 70 validation), fixed manifest split, seed 0. All model inputs are FPS-, duration-, and timestamp-blind. The ±0.3-second metric below is computed only after frame-number predictions, using manifest FPS for offline evaluation; model selection uses a FPS-blind 1%-of-clip score. No leaderboard result is claimed.

## Controlled design

B0 is the existing 128-position ASFormer checkpoint (1,038,952 trainable parameters). P1–P4 share a 273,942-parameter 1D pyramid: 70×384 frozen DINO grid features → per-token 384→16 projection → 1120→128 frame projection → four stride-2 levels with residual depthwise-kernel-5/pointwise blocks → linear top-down fusion → dense ENTRY/COLLISION at the finest level and pooled side/evasion heads. Dropout 0.35. No FPS or duration is read by the dataset or head, and decoding maps normalized positions back to original frame numbers while enforcing ENTRY ≤ COLLISION.

P1 always requests 128 positions. P2–P4 request all available frames for N≤128; 128 for 129–256; 192 for 257–512; 256 for 513–768; and 320 for N>768. P3 adds ±25% count variation, grid jitter, 2.5% frame drop, and a mild monotonic warp during training. P4 adds two-view CDF-L1 consistency at weight 0.05. All use the same frozen checkpoint and normalized Gaussian event targets (width 0.015), AdamW at 0.001, weight decay 0.05, one-cycle schedule, batch 4, up to 30 epochs, early-stop patience 7. The best FPS-blind validation checkpoint, not final weights, is evaluated.

The 383-grid cache reused the old 192-grid features and encoded 17,329 additional unique frames. Its provenance is in `stage2/temporal_pyramid_experiments/cache_383/meta.json`; old cache/submission files were not changed.

## Overall validation

| Run | Score | ENTRY | COLLISION | Side F1 | Evasion F1 | Entry normalized MAE | Collision normalized MAE | Parameters |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.689 | 0.557 | 0.814 | 0.828 | 0.568 | 0.070 | 0.025 | 1,038,952 |
| P1 | 0.674 | 0.571 | 0.743 | 0.840 | 0.585 | 0.073 | 0.058 | 273,942 |
| P2 | 0.730 | 0.629 | 0.786 | 0.871 | 0.694 | 0.057 | 0.039 | 273,942 |
| P3 | 0.690 | 0.614 | 0.757 | 0.828 | 0.569 | 0.052 | 0.034 | 273,942 |
| P4 | 0.703 | 0.629 | 0.757 | 0.871 | 0.583 | 0.044 | 0.030 | 273,942 |

## Frame-count bins

Each cell is score (n); per-bin macro-F1 at n≤2 is highly unstable. There are no validation clips in 750–1000.

| Run | <150 | 150–300 | 300–500 | 500–750 | 750–1000 | >1000 | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.780 (32) | 0.686 (21) | 0.450 (2) | 0.850 (1) | — (0) | 0.492 (14) | 0.689 |
| P1 | 0.749 (32) | 0.685 (21) | 0.725 (2) | 0.850 (1) | — (0) | 0.442 (14) | 0.674 |
| P2 | 0.802 (32) | 0.790 (21) | 0.725 (2) | 0.500 (1) | — (0) | 0.463 (14) | 0.730 |
| P3 | 0.759 (32) | 0.738 (21) | 0.625 (2) | 0.150 (1) | — (0) | 0.488 (14) | 0.690 |
| P4 | 0.786 (32) | 0.745 (21) | 0.475 (2) | 0.425 (1) | — (0) | 0.484 (14) | 0.703 |

Detailed metrics for every nonempty bin:

| Run | Bin | n | ENTRY | COLLISION | Side F1 | Evasion F1 | Score | Entry norm MAE | Collision norm MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | <150 | 32 | 0.688 | 0.906 | 0.873 | 0.610 | 0.780 | 0.073 | 0.018 |
| B0 | 150–300 | 21 | 0.571 | 0.762 | 0.905 | 0.555 | 0.686 | 0.061 | 0.020 |
| B0 | 300–500 | 2 | 0.000 | 1.000 | 0.333 | 0.333 | 0.450 | 0.067 | 0.005 |
| B0 | 500–750 | 1 | 1.000 | 1.000 | 0.500 | 0.500 | 0.850 | 0.006 | 0.009 |
| B0 | 750–1000 | 0 | — | — | — | — | — | — | — |
| B0 | >1000 | 14 | 0.286 | 0.643 | 0.641 | 0.475 | 0.492 | 0.083 | 0.053 |
| P1 | <150 | 32 | 0.688 | 0.875 | 0.762 | 0.584 | 0.749 | 0.072 | 0.038 |
| P1 | 150–300 | 21 | 0.524 | 0.762 | 1.000 | 0.568 | 0.685 | 0.033 | 0.024 |
| P1 | 300–500 | 2 | 0.500 | 1.000 | 0.333 | 1.000 | 0.725 | 0.032 | 0.015 |
| P1 | 500–750 | 1 | 1.000 | 1.000 | 0.500 | 0.500 | 0.850 | 0.009 | 0.006 |
| P1 | 750–1000 | 0 | — | — | — | — | — | — | — |
| P1 | >1000 | 14 | 0.357 | 0.357 | 0.785 | 0.497 | 0.442 | 0.143 | 0.161 |
| P2 | <150 | 32 | 0.656 | 0.969 | 0.875 | 0.681 | 0.802 | 0.059 | 0.019 |
| P2 | 150–300 | 21 | 0.714 | 0.810 | 0.952 | 0.762 | 0.790 | 0.024 | 0.024 |
| P2 | 300–500 | 2 | 0.500 | 1.000 | 0.333 | 1.000 | 0.725 | 0.033 | 0.007 |
| P2 | 500–750 | 1 | 1.000 | 0.000 | 0.500 | 0.500 | 0.500 | 0.002 | 0.020 |
| P2 | 750–1000 | 0 | — | — | — | — | — | — | — |
| P2 | >1000 | 14 | 0.429 | 0.357 | 0.775 | 0.475 | 0.463 | 0.112 | 0.114 |
| P3 | <150 | 32 | 0.656 | 0.938 | 0.779 | 0.559 | 0.759 | 0.074 | 0.033 |
| P3 | 150–300 | 21 | 0.714 | 0.714 | 0.952 | 0.637 | 0.738 | 0.025 | 0.028 |
| P3 | 300–500 | 2 | 0.500 | 1.000 | 0.333 | 0.333 | 0.625 | 0.035 | 0.003 |
| P3 | 500–750 | 1 | 0.000 | 0.000 | 0.500 | 0.500 | 0.150 | 0.026 | 0.019 |
| P3 | 750–1000 | 0 | — | — | — | — | — | — | — |
| P3 | >1000 | 14 | 0.429 | 0.429 | 0.775 | 0.475 | 0.488 | 0.044 | 0.051 |
| P4 | <150 | 32 | 0.719 | 0.938 | 0.779 | 0.593 | 0.786 | 0.049 | 0.021 |
| P4 | 150–300 | 21 | 0.714 | 0.714 | 0.952 | 0.679 | 0.745 | 0.024 | 0.037 |
| P4 | 300–500 | 2 | 0.000 | 0.500 | 1.000 | 1.000 | 0.475 | 0.040 | 0.020 |
| P4 | 500–750 | 1 | 1.000 | 0.000 | 0.500 | 0.000 | 0.425 | 0.009 | 0.019 |
| P4 | 750–1000 | 0 | — | — | — | — | — | — | — |
| P4 | >1000 | 14 | 0.357 | 0.500 | 0.928 | 0.300 | 0.484 | 0.067 | 0.043 |

## Sources and confounding

| Run | AIHub (18) | CCD (17) | MM-AU (20) | Nexar (15) |
|---|---:|---:|---:|---:|
| B0 | 0.663 | 0.696 | 0.815 | 0.522 |
| P1 | 0.658 | 0.672 | 0.815 | 0.480 |
| P2 | 0.775 | 0.705 | 0.895 | 0.473 |
| P3 | 0.706 | 0.673 | 0.829 | 0.473 |
| P4 | 0.709 | 0.686 | 0.859 | 0.486 |

Nexar event hits (15 clips):

| Run | ENTRY hits | COLLISION hits | Entry norm MAE | Collision norm MAE |
|---|---:|---:|---:|---:|
| B0 | 5/15 | 10/15 | 0.078 | 0.051 |
| P1 | 6/15 | 6/15 | 0.134 | 0.151 |
| P2 | 7/15 | 5/15 | 0.105 | 0.108 |
| P3 | 6/15 | 6/15 | 0.043 | 0.049 |
| P4 | 6/15 | 7/15 | 0.063 | 0.041 |

There is almost no source/length overlap: validation AIHub clips are all 150 frames, CCD all 50, MM-AU spans 50–355 (median 102), and Nexar spans 540–1248 (median 1210). CausalCrash has one training clip and no validation clip. A within-source frame-count effect cannot be estimated reliably for three nearly fixed-length sources. Nexar has only 15 validation examples. Therefore the MM-AU/Nexar gap cannot be numerically decomposed into length versus domain shift on this split.

Spearman correlation between frame count and a per-video correctness proxy (event hits plus attribute correctness, not macro-F1):

| Run | Global rho | Nexar rho | Global entry-error rho | Global collision-error rho |
|---|---:|---:|---:|---:|
| B0 | -0.320 | -0.109 | -0.113 | 0.080 |
| P1 | -0.277 | -0.112 | -0.113 | 0.177 |
| P2 | -0.250 | -0.360 | -0.187 | 0.066 |
| P3 | -0.239 | -0.258 | -0.268 | -0.083 |
| P4 | -0.284 | -0.518 | -0.302 | 0.009 |

Full source-specific correlations are saved in `stage2/temporal_pyramid_experiments/results/analysis.json`. The plotted bin scores and validation curves are in [framecount_and_validation.png](../stage2/temporal_pyramid_experiments/results/framecount_and_validation.png). Dashed training curves are available for P1–P4; B0's historical run logged validation but not train scores each epoch.

## Overfitting and resource cost

| Run | Best epoch | Last epoch | Gap (epochs) | Train selection score at best weights | Val selection score | Head ms/video | Peak allocated VRAM MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 14 | 30 | 16 | — | 0.464 | 0.157 | 616.4 |
| P1 | 11 | 18 | 7 | 0.870 | 0.444 | 0.660 | 207.7 |
| P2 | 7 | 14 | 7 | 0.716 | 0.475 | 0.636 | 489.4 |
| P3 | 6 | 13 | 7 | 0.607 | 0.455 | 0.642 | 484.3 |
| P4 | 5 | 12 | 7 | 0.654 | 0.453 | 0.614 | 830.1 |

P1–P4 continue improving training metrics after validation peaks, confirming substantial overfitting despite the smaller head. In particular P1's best-weight train/validation selection scores are 0.870/0.444; P2's are 0.716/0.475. Early stopping is essential.

## Sampling-density robustness

For each video, predictions were compared at 96/128/192 positions (B0/P1) or 96/128/192/256/320 positions (P2–P4). Counts above the old 192 cache are not treated as independent encoded frames. Short videos are capped at native count for adaptive models.

| Run | Score range | Entry normalized prediction SD | Collision normalized prediction SD | Entry mean frame range | Collision mean frame range |
|---|---:|---:|---:|---:|---:|
| B0 | 0.028 | 0.0184 | 0.0082 | 17.6 | 9.3 |
| P1 | 0.055 | 0.0182 | 0.0185 | 24.1 | 17.4 |
| P2 | 0.028 | 0.0131 | 0.0102 | 27.8 | 19.4 |
| P3 | 0.024 | 0.0077 | 0.0087 | 8.5 | 6.7 |
| P4 | 0.033 | 0.0094 | 0.0036 | 20.7 | 4.7 |

P3 lowers ENTRY view sensitivity versus P2, while P4 lowers COLLISION view sensitivity; neither produces a higher overall score. P4's score range is not lower than P3's, so the consistency loss has no clear net robustness benefit.

A single long Nexar raw-frame benchmark using the same frozen DINO backbone and batch size 12 took 1.12 s for 128 unique frames and 2.00 s for 320, including frame loading/preprocessing and DINO feature extraction. The peak PyTorch-allocated VRAM was 405 MiB in both cases; CUDA context/reservation is additional. This is one warm-file benchmark under concurrent Stage 3 activity, not a competition runtime guarantee. Across validation, adaptive sampling encodes 143.7 unique frames/video versus 100.8 for fixed-128 deduplicated sampling (+42.6%); for >768-frame clips the cost is 320 versus 128 (2.5×).

## Failure cases and explicit conclusions

P2 misses long Nexar COLLISION by selecting unrelated peaks: `nexar_00200` predicts frame 113 versus GT 591, `nexar_00763` 1132 versus GT 591, and `nexar_00927` 1209 versus GT 608. Finer sampling did not prevent false-event selection. B0 also has large ENTRY misses on some of these clips. Details are in each run's `predictions.json`.

- Is frame count independently causal? **Not identifiable** on this split; source and length have minimal overlap. The global score/length correlation is negative, but within Nexar it is unstable and sometimes reverses.
- MM-AU versus Nexar after controlling for frame count? **Cannot estimate** without overlapping lengths or within-source length variation.
- Pyramid at fixed 128? **No overall gain:** P1 0.674 versus B0 0.689; long COLLISION falls from 10/15 to 6/15.
- Adaptive sampling on long ENTRY? **Yes, narrowly:** Nexar ENTRY 5/15→7/15 versus B0, but only 14 >1000-frame clips and no replication.
- Adaptive sampling on long COLLISION? **No:** Nexar 10/15→5/15, outweighing the ENTRY gain for long clips.
- Short-video harm? **No in P2:** <150 score 0.780→0.802; 150–300 score 0.686→0.790. Most of P2's overall gain comes from short/mid-length sources.
- Temporal augmentation? **Some prediction stability, not score:** P3 overall 0.690; long score 0.488 versus P2 0.463.
- Cross-view consistency? **Mixed:** P4 COLLISION prediction SD falls, but its overall score is 0.703 and density score range rises versus P3.
- Flattest long-vs-short score curve? None solves the gap; B0 has the best >1000 score (0.492), with P3 close (0.488).
- Replace current Stage 2 head? **No.** P2 has the best overall validation score (0.730) but worsens the key long-video/Nexar COLLISION failure and increases raw-frame cost. Single-seed differences on 70 validation videos do not establish leaderboard generalization.

Best controlled experiment: **P2**, overall **0.730**, ENTRY **0.629**, COLLISION **0.786**, 273,942 trainable parameters, head **0.636 ms/video** plus frozen DINO, checkpoint `stage2/temporal_pyramid_experiments/results/P2/seed0/checkpoint.pt`. The temporal-pyramid-as-a-long-video-solution hypothesis is **not supported**, although P2 improved the aggregate score.
