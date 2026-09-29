# GPU seed3 matched replication

Five known-duplicate-grouped out-of-fold pairs, 349 held-out clips and 338 known-duplicate groups. Baseline and sharp auxiliary ENTRY used the same GPU, seed3, splits, stride augmentation, and FPS-blind checkpoint selection. Scores below are from CPU re-inference of the saved checkpoints at three sampling rates. The candidate was selected after the seed0 screen; intervals are descriptive paired group bootstraps, not search-adjusted.

| Sampling rate | Baseline | Sharp auxiliary | Sharp − baseline | 95% paired group interval |
|---|---:|---:|---:|---:|
| Native | 0.740217 | 0.732501 | −0.007716 | [−0.029692, +0.014427] |
| Half | 0.696645 | 0.692788 | −0.003857 | [−0.026357, +0.017936] |
| Third | 0.637489 | 0.647488 | +0.009999 | [−0.014769, +0.034659] |

NEXAR sharp-minus-baseline deltas were −0.034715/−0.043525/−0.011068 at native/half/third rate; >1000-frame deltas were −0.022801/−0.036387/−0.002712. Fold-level signs varied. Four pairs had exact saved-GPU to CPU native decoded prediction parity; fold1 sharp had one adjacent ENTRY frame difference on `ccd_001362` (18 saved versus 17 CPU). No other mismatches passed the bounded evaluator. The full details, source breakdowns, fold rows, and interval computation are in `gpu_seed3_aggregate.json` and `aggregate_gpu_seed3.py`.

**Decision:** retain v8. Sharp auxiliary ENTRY has no confirmed benefit on this independent seed, and source weakness on NEXAR conflicts with the target of robust real-leaderboard performance. Do not convert these grouped single-model validation scores into a numerical leaderboard forecast. Wait for the independently running CPU seeds1/2 and examine matched per-source/rate behavior before any ensemble test or deployment change.
