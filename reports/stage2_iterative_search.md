# Stage 2 iterative search — 2026-09-23

Experiment window requested by user: approximately 23:12–02:12 UTC. Goal: 0.80 offline competition-style score on the unchanged 70-video validation split, without FPS/duration/timestamp as model inputs and without interfering with Stage 3. This is a research target, not a promise of leaderboard generalization. All checkpoints, predictions, histories, configs, and metrics remain in isolated `stage2/temporal_pyramid_experiments/results/<run>/seed*` directories.

## Current evidence

| Run | Change from P2 | Offline score | ENTRY | COLLISION | Decision |
|---|---|---:|---:|---:|---|
| B0 | Existing ASFormer, fixed 128 | 0.689 | 0.557 | 0.814 | Baseline; best long COLLISION |
| P2 seed 0 | Four-level compact pyramid, adaptive K | 0.730 | 0.629 | 0.786 | Best single seed so far |
| P5 global coarse | Global attention at L3 | 0.696 | 0.614 | 0.743 | Reject: worse |
| P6 long oversample | All train clips once + extra >500-frame draws | 0.677 | 0.600 | 0.714 | Reject: worse |
| P7 event categories | Event-conditioned side/evasion heads | 0.710 | 0.643 | 0.771 | Reject as whole model: attributes fell |
| A1 ASFormer adaptive | Original larger temporal head, adaptive K | 0.535 | 0.371 | 0.543 | Reject: severe overfit |
| P2 seed 1 | Independent initialization/shuffle | 0.703 | 0.571 | 0.743 | Variation confirmed |
| P2 seed 2 | Independent initialization/shuffle | 0.688 | 0.543 | 0.771 | Variation confirmed |
| P2 seed 3 | Independent initialization/shuffle | 0.729 | 0.586 | 0.800 | Variation confirmed |
| ENS_P2_seed0123 | Equal probability average of four P2 seeds | **0.752** | 0.629 | 0.829 | Best aggregate so far |
| P8 hard negative | +0.1 ranking penalty on distant false peaks | 0.693 | 0.600 | 0.757 | Reject: no event gain |
| P9 adaptive target | Cap dense target width at two sampled positions | 0.689 | 0.586 | 0.771 | Reject: no gain |
| P10 EMA 0.99 seed 0 | EMA-smoothed checkpoint selection | 0.741 | 0.671 | 0.786 | Promising single-head gain; replication underway |
| GAP_ENS_P2_0123 | Train-fitted sampled-index gap prior on four-seed ensemble | 0.742 | 0.600 | 0.829 | Reject: loses ENTRY hits |

The four P2 single-seed scores span 0.688–0.730 (mean 0.713). The ensemble gain is plausible variance reduction, but the fixed validation set has only 70 clips. No model has demonstrated leaderboard improvement. The original temporal-pyramid report is [stage2_temporal_pyramid_framecount_experiments.md](stage2_temporal_pyramid_framecount_experiments.md).

## Iteration plan and guardrails

1. Replicate P10 EMA on another seed to determine whether its ENTRY gain is stable.
2. Test mild spatial-feature regularization (token dropout and channel jitter), keeping architecture/sampling/loss fixed.
3. Compare each variant overall, by frame-count bin, by source, and specifically NEXAR ENTRY/COLLISION. Only keep a variant if long-video behavior and robustness improve, not just one aggregate score.
4. If useful, validate promising variants across seeds and with a predeclared probability ensemble. Avoid validation-specific per-source or frame-count gates; inference may use frame count only for sampling density.
5. Recheck GPU headroom and keep the existing Stage 3 processes untouched. All training uses an allocator cap of 5% of the 32 GiB GPU and starts only with ≥3 GiB free.

The 0.80 target is particularly difficult because B0 and P2 make complementary event errors: an *oracle* choosing the correct model per event would reach 52/70 ENTRY and 61/70 COLLISION hits, but those choices are not known at inference. Any learned selector would need out-of-fold training; fitting a gate on the 70 validation labels would be invalid.
