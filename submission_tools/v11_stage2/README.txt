Stage 2 v11 (robust): E4 / E2 (boundary head) / XN4 (+670 NEXAR) AuxPyramid families, each trained with
  * temporal-rate augmentation, equal mix of 1/1, 1/2, 1/3 of the native frame rate (motion recomputed on retained frames),
  * 629 metadata-anchored MM-AU / CCD extras (near-duplicates of labelled clips removed; extras stride-augmented too),
  * unlabelled cross-frame-rate consistency (weight 0.3; 1,058-clip unlabelled pool, native view = teacher).
4 seeds each (0-3), full-data refit on all 349 labelled clips; stop epochs = CV medians (E4 10, E2 8, XN4 15). Runtime identical to v8/v10.
Selection (reports/stage2_robustness_campaign.md): best ensemble at 1/3 frame rate, the validation view that ranked v5-v8 correctly
on the leaderboard: duplicate-clean CV 0.736 at 1/3 rate (v10 0.729, v8 recipe 0.710); native 0.758 (v10 0.768); NEXAR @1/3 0.692.
Stage 3: V3 unchanged, steering 7.0 deg / acceleration +-0.4 (same as v10, so v10 -> v11 isolates the Stage 2 change). Stage 1 unchanged.
