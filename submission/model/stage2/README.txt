Stage 2: P2 compact temporal pyramid (4 stride-2 levels, 274k params), equal-weight probability ensemble of seeds 0-3,
on frozen geometry-tuned (no-anchor) DINOv3-S 7x10 pooled patch tokens.
Sampling depends only on the number of available frames: all frames for N<=128; 128/192/256/320 positions above,
taken from FPS-blind 192- or 383-position normalized grids. Entry <= collision constrained decoding.
members/p2_seed{0..3}.pt: stage2/temporal_pyramid_experiments/results/REFIT_P2_all349/seed{0..3}/checkpoint.pt
  (full refit on all 349 clips; epochs 7/8/6/13 = each seed's validation-selected epoch; no held-out validation).
Validated counterpart (P2 seeds 0-3 trained on 279 clips): 0.757 on 70 validation clips (batch 1).
backbone.pth: outputs/geometry_pretrain/runs/phase1_partial_noanchor/backbone_best.pth
No FPS or duration metadata is read.
