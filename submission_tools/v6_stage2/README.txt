Stage 2 v6 (event-specific ensemble, stage2/aux_signal_experiments/event_select.py; log event_select_7.out):
18 heads on frozen geometry-tuned (no-anchor) DINOv3-S 7x10 pooled patch tokens, all trained on the 279-clip train split.
  C0_avg seed0-1, X_ema seed0-1       LCPyramid, P2 recipe (X_ema: EMA 0.99)
  M_motion seed0-1                    LCPyramid + 28-d global camera-motion input
  PH_phase seed0-1                    LCPyramid, phase-loss recipe, direct heads
  E4_NT_both seed0-2                  AuxPyramid, NT loss, 100-d motion input (28 global + 72 residual)
  E2_NT_both_bnd2 seed0-2             AuxPyramid, as E4 + boundary [h_t, dh_t] auxiliary loss (head unused at inference)
  M1_both seed0-3                     AuxPyramid, P2 loss, 100-d motion input
Residual motion: similarity camera motion (sparse Lucas-Kanade + RANSAC) removed, residual summarised per native frame (CPU).
ENTRY    = mean probability of E2_NT_both_bnd2, M1_both, PH_phase, E4_NT_both, M_motion members
COLLISION = mean probability of E2_NT_both_bnd2 members + 0.5 * robust-z(max camera shift within +-half spacing), native snap
side / evasion = mean over all 18 members, threshold 0.5.
Member counts per family equal the CV seed counts of the evaluated ensemble.
5-fold CV (349 clips, 80 NEXAR, family subsets and beta chosen on other folds): overall 0.7884, NEXAR 0.7103, >1000 0.6933
(v5: 0.769 / 0.670). No FPS, duration or timestamps are read.
