# Stage 2 deployment training method: nested-fold ensemble versus full refit

**Decision: reject the fold-trained deployment ensemble.** This one-fold screen does not support replacing v8's fixed-epoch full-data refit. The active leaderboard goal remains unmet; the best verified Stage 2 score is v8's 0.59293.

## Protocol

The outer grouped fold0 has 276 training clips and 73 validation clips. Known-duplicate exclusions leave 54 outer validation clips (AIHUB 11, CCD 14, MMAU 13, NEXAR 16). No outer validation label is used in model training or selection. Four E4_sa controls (seeds 0–3) train on all 276 outer training clips and stop at epoch7, the locked v8 full-refit epoch. Four candidate models use disjoint inner holdouts (grouped folds1–4); each trains on the other three quarters and selects its checkpoint only on its inner holdout. The two recipes have four heads each, the same frozen features, motion inputs, stride mix (0.5/0.25/0.25), probability averaging, and direct decoder. This controls ensemble size and inference compute; the smaller training set per inner head is part of the cross-fitting method's trade-off.

The predeclared gate in `manifest.json` required at least +0.01 native and third-rate score on this duplicate-clean outer fold, with no source loss above 0.02. This was a screening gate, not a statistical significance test.

| Outer-fold score | Full refit control | Inner-fold ensemble | Inner minus full |
|---|---:|---:|---:|
| Native | 0.7462 | 0.7317 | −0.0145 |
| Half rate | 0.7445 | 0.7187 | −0.0258 |
| Third rate | 0.7450 | 0.7159 | −0.0292 |

Paired 1,500-resample clip bootstrap intervals for inner minus full are [−0.0559, +0.0227] native, [−0.0722, +0.0170] half, and [−0.0838, +0.0232] third. The intervals include zero because the outer sample is small, but all three observed deltas oppose the hypothesis and fail the gate. At third rate, ENTRY accuracy is 0.5741 → 0.5370, COLLISION 0.8889 → 0.8704, and NEXAR score 0.7031 → 0.6278. Do not expand to other outer folds or package this candidate.

The comparison only covers the E4 family, not the full E4+E2+XN4 submission. It does show that checkpoint selection through inner folds is not an obvious remedy for the validation-to-leaderboard gap; losing training clips per head likely costs more than selection helps here. The v8 package's previously verified inference parity remains relevant, so changing its deployment recipe on this evidence would add cost and risk without a measured benefit.

Artifacts: `crossfit_deploy.py`, `manifest.json`, `inner{1..4}_{train,val}.jsonl`, eight isolated checkpoint directories under `full/` and `inner/`, `result.json`, and per-clip predictions for each rate. The nested runner waited while three other GPU cache jobs were active; it did not modify or stop those jobs.
