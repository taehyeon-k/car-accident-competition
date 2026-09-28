Stage 2 v13 (robust, no labelled MM-AU/CCD extras): v8's three families (E4, E2 boundary head, XN4 with v8's 670 NEXAR extras),
each trained with
  * temporal-rate augmentation, equal mix of 1/1, 1/2, 1/3 of the native frame rate (v8: 0.5 / 0.25 / 0.25),
  * unlabelled cross-frame-rate consistency (weight 0.3; 1,046-clip unlabelled pool, no labels used),
  * window-crop augmentation (p 0.3: random windows containing ENTRY..COLLISION; fixes the truncation / position-prior failure).
No metadata-/teacher-labelled MM-AU/CCD clips (the ingredient of v10 / v12, both below v8 on the LB).
4 seeds each (0-3), full-data refit on all 349 labelled clips; stop epochs = CV medians (E4 9, E2 8, XN4 8). Runtime identical to v8.
Stage 3: V3 unchanged, steering 7.0 deg / acceleration +-0.4 (best measured Stage 3, v10: 0.7475). Stage 1 unchanged.
