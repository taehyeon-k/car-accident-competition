Stage 2 (long-context v2): equal-weight probability ensemble of 15 LCPyramid heads (274-294k params each) on frozen
geometry-tuned (no-anchor) DINOv3-S 7x10 pooled patch tokens, + camera-shift motion fusion for COLLISION + native snap.
Members (all trained on the 279-clip train split of the 349 labelled clips; NOT refit on all 349):
  C0_avg seed0-3        P2 recipe, direct heads
  X_ema seed0-3         P2 recipe + EMA 0.99
  M_motion seed0-2      P2 recipe + motion input (28-dim max/mean of per-frame shift/diff descriptors)
  PH_phase seed0-3      phase-based temporal loss (stage2/phase_loss.md), decoded with its direct heads
Decoding: argmax over entry<=collision of log p_entry + [log p_collision + 1.0 * robust-z(max global shift within +-half sample
spacing)], then collision snapped to the native frame with the largest shift within +-half spacing. Attributes: ensemble-mean heads.
Validation (fixed 70-clip val split, batch 1): overall 0.767, NEXAR (15) 0.619, >1000 frames 0.594
  (current v3/v4 P2 ensemble on the same split: 0.757 / 0.500 / 0.492). 5-fold CV (349 clips, 80 NEXAR): overall 0.769, NEXAR 0.670.
backbone.pth = outputs/geometry_pretrain/runs/phase1_partial_noanchor/backbone_best.pth. No FPS, duration or timestamps are read.
Source: stage2/long_context_v2_experiments (commit 28f2c0d); reports/stage2_phase_loss_report.md.
