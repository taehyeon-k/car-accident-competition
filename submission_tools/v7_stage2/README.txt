Stage 2 v7: equal-weight probability ensemble of 12 AuxPyramid heads on frozen geometry-tuned (no-anchor) DINOv3-S 7x10 tokens,
each with a 100-d motion input (28 global camera-motion + 72 ego-motion-compensated residual-motion dims, CPU, no RAFT):
  E4_full  seed0-3   NT loss (direct Gaussian targets, attributes 0.5/0.5), 349 labelled clips, stop epoch 8 (CV median best)
  E2_full  seed0-3   as E4 + boundary [h_t, dh_t] auxiliary loss (head unused at inference), stop epoch 8
  XN4_full seed0-3   as E4 + 670 unlabelled NEXAR clips (COLLISION from Nexar's public time_of_event + offset from the labelled
                     NEXAR clips; ENTRY pseudo-labels from the full-data E4+E2 teacher within 2 s before it, weight 0.5;
                     attributes masked), stop epoch 9
Decoding (v5): ENTRY and COLLISION = mean over all 12 members; COLLISION + 0.5 * robust-z(max camera shift within +-half spacing),
argmax over entry <= collision, native-frame snap; side / evasion = member mean, threshold 0.5.
5-fold CV of this recipe (349 clips, 80 NEXAR, members trained per fold): overall 0.7995, NEXAR 0.721, >1000 0.709
(v5: 0.769 / 0.670). Runtime: submission_tools/v6_stage2 (parity-checked against training). No FPS, duration or timestamps.
Stage 3: V3 unchanged, decoder overrides steering 3.0 deg, acceleration +-0.5 m/s^2. Stage 1: unchanged.
