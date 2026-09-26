Stage 2 v9 (generalization): the v7 recipe (E4 + E2 + XN4 residual-motion AuxPyramid heads, 4 seeds each, full-data refit) trained with
(1) temporal-rate augmentation (each labelled clip at 1/1, 1/2, 1/3 frame rate, p = .5/.25/.25, motion recomputed) and (2) EMA weights
(decay 0.99; the EMA copy is what is saved). Training-only changes; runtime identical to v7/v8. Stop epochs (CV medians): 8 / 8 / 12.
Evidence (reports/stage2_generalization_research.md): leave-one-source-out, E4+E2 ensemble: mean 0.698 / worst source 0.593 vs
0.672 / 0.564 (stride aug only) and 0.663 / 0.561 (v7 E4); improves all 4 unseen sources. 5-fold CV of the recipe 0.783 / NEXAR 0.707
(v8 recipe 0.787 / 0.707). Frame-rate robustness 0.771 / 0.747 / 0.696 at 1/1, 1/2, 1/3 fps (v8 0.775 / 0.759 / 0.718, v7 0.772 / 0.720 / 0.648).
Stage 3: V3 unchanged, steering 3.0 deg / acceleration +-0.5 (as v7/v8). Stage 1: unchanged.
