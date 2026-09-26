Stage 2 v8 (robust): the v7 recipe (E4 + E2 + XN4, residual-motion AuxPyramid heads, 4 seeds each, full-data refit) trained with
TEMPORAL-RATE AUGMENTATION: every labelled clip is seen at 1/1, 1/2 or 1/3 of its native frame rate (p = .5/.25/.25; adaptive
re-sampling and global/residual motion recomputed on the retained frames). Training-only change; runtime identical to v7/v6.
Stop epochs (CV medians of the augmented families): E4_sa 7, E2_sa 8, XN4_sa 12.
Evidence (reports/stage2_generalization_research.md): 5-fold CV of the recipe 0.787 / NEXAR 0.707 vs v7 recipe 0.784 / 0.717 (same
seeds, v5 decoding); frame-rate robustness (plain decoding) 0.775 / 0.759 / 0.718 at 1/1, 1/2, 1/3 fps vs 0.772 / 0.720 / 0.648.
Stage 3: V3 unchanged, steering 3.0 deg / acceleration +-0.5 (as v7, so v8 vs v7 isolates the Stage 2 change). Stage 1: unchanged.
