Stage 2 v10: the v8 recipe (E4 / E2 / XN4 AuxPyramid heads with global+residual motion, temporal-rate augmentation 1/1, 1/2, 1/3 at
p = .5/.25/.25) trained with 658 extra MM-AU / CCD clips labelled from public metadata (MM-AU t_co / t_ai, CCD accident frame; ENTRY from
a full-data E4_sa teacher; 81 near-duplicates of labelled clips removed; extras stride-augmented too; attributes masked):
  XC_E4_full   E4 + extras supervising COLLISION only            stop epoch 9
  XS_E2_full   E2 (boundary head) + extras (COLLISION + ENTRY)   stop epoch 9
  XS_XN4_full  E4 + 670 NEXAR extras + MM-AU/CCD extras         stop epoch 15
4 seeds each (0-3), full-data refit on all 349 labelled clips; stop epochs = CV medians. Runtime identical to v8 (same code).
Selection evidence (duplicate-clean CV, 284 clips, 3 seeds/family, plain decoding; reports/stage2_leaderboard_correlation.md):
  CV at 1/3 frame rate (the view that ranks v5-v8 correctly on the LB): 0.729 vs v8 recipe 0.710; native 0.768 vs 0.764; 1/2 rate 0.762 vs 0.750.
  Caveat: chosen among 4 ensemble mixes on the same CV; +0.018 at 1/3 rate is mildly optimistic.
Stage 3: V3 unchanged, decoder thresholds steering 7.0 deg / acceleration +-0.4 (user request). Stage 1: unchanged.
