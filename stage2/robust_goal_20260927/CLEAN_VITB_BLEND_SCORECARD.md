# Clean ViT-B addition to v8: duplicate-clean CV screen

Fixed equal-weight probability average, three seeds per family, five original CV folds, 284 duplicate-clean held-out clips. All three recipes use the same plain decoder and cached input views; 2,000 paired known-duplicate-group bootstrap samples per rate. The additional ViT-B head `E4_sbB` was trained without MM-AU/CCD pseudo labels. The v8 baseline retains its existing NEXAR expansion. This is an offline CV proxy, not the scored full-data v8 package.

| Rate | v8 plain | + ViT-S E4_sb control | + ViT-B E4_sbB | ViT-B − v8 [95% paired interval] |
|---|---:|---:|---:|---:|
| Native | 0.764441 | 0.767728 | 0.767891 | +0.003449 [−0.007358, +0.014762] |
| Half | 0.749563 | 0.754054 | 0.755040 | +0.005477 [−0.003979, +0.015881] |
| Third | 0.710388 | 0.711759 | 0.715111 | +0.004723 [−0.005576, +0.015573] |

The ViT-B-minus-ViT-S control deltas are only +0.000163/+0.000986/+0.003351 at native/half/third rate, each with an interval spanning zero. ViT-B-minus-v8 NEXAR is +0.005724/+0.010698/**−0.016786**; >1000-frame clips are +0.004077/+0.000048/**−0.017518**. Thus the small aggregate third-rate gain hides the hardest-source regression. Data: `clean_vitb_blend_results.json`; reproducible uncertainty and source table: `clean_vitb_blend_scorecard.json`, `score_clean_vitb_blend.py`.

**Decision:** do not replace or enlarge v8 for this small and inconsistent gain, which is far below the 0.63–0.65 real-leaderboard target and may increase inference cost through a second backbone. Do not project the CV delta onto LB: v10 falsified the earlier third-rate calibration.
