# Experiment analysis — September 15, 2026

## Recommendation

Prioritize consistent evaluation and better decoding before another broad learning-rate/dropout sweep. The two new training runs deliver modest aggregate gains, but neither improves the combined event accuracy at its selected checkpoint. A CPU replay of the old baseline's saved logits yields a larger improvement from changing the decision rule alone. That result is exploratory and needs confirmation on the new checkpoints and additional held-out data.

## Scope and evidence

Snapshot cutoff: **2026-09-15 07:12 UTC**. Examined the two overnight runs and the preceding baseline. The dropout run had 14 complete validation epochs and no exit record at the cutoff; its results are provisional. It was not stopped or resumed by this analysis.

Sources: local W&B binary records with checksum verification, both best/last checkpoint metadata for all three runs, submission checkpoint metadata, current and packaged inference code, baseline per-video errors and saved logits, manifests, and the saved inference performance profile. No new backbone inference or training was run. No leaderboard scores were available in the inspected files. The current workspace contains two exported submissions, not evidence that either was accepted or scored by the competition.

### Actual configurations

These values come from the saved checkpoints, not the current YAML or the run name.

| Setting | Baseline `kqr8dczz` | `span_lr` `irfg2z4m` | `span_lr_dr` `15ov22sn` |
|---|---:|---:|---:|
| Start, UTC | Sep 14 16:43 | Sep 15 01:09 | Sep 15 04:22 |
| DINO LoRA LR | 2e-5 | 4e-6 | 5e-6 |
| V-JEPA LoRA LR | 1e-5 | 3e-6 | 5e-6 |
| Head LR | 3e-4 | 3e-4 | 3e-4 |
| ENTRY target sigma, seconds | 0.10 | 0.20 | 0.20 |
| COLLISION target sigma, seconds | 0.10 | 0.20 | 0.10 |
| Spatial / temporal dropout | 0.10 / 0.10 | 0.10 / 0.10 | 0.15 / 0.15 |
| Attribute / LoRA dropout | 0.15 / 0.05 | 0.15 / 0.05 | 0.30 / 0.10 |
| Early-stopping patience | 6 | 4 | 4 |
| Configured validation span | absent | absent | absent |
| Training frame cap | 512 | 512 | 512 |

All retain the 256-D head, nominal effective batch size 4, seed 42, the same manifest paths and 70/15/15 temporal augmentation probabilities. Run two changed LR and both target widths together; run three changed four dropout rates, both backbone LRs and collision target width together. These runs cannot isolate the causal effect of dropout, learning rate or smoothing.

### Best logged results

| Run | Best epoch | Score | ENTRY | COLLISION | Side macro F1 | Evasion macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 13 | 0.6616 | 50% | 80% | 0.8182 | 0.5593 |
| `span_lr` | 14 | 0.6678 | 54% | 76% | 0.7987 | 0.6198 |
| `span_lr_dr`, provisional | 12 | 0.6770 | 52% | 78% | 0.8400 | 0.6400 |

**All three have 65 correct event predictions out of 100 ENTRY/COLLISION decisions.** Because the two events have equal competition weight, their total event contribution is exactly 0.455 in all three selected checkpoints. The aggregate score gains come from the attributes. This does not mean all runs succeed on the same videos; newer per-video predictions are needed to measure overlap.

The latest run improves only 0.0154 over baseline and 0.0092 over `span_lr`. One changed event decision among 50 videos contributes 0.007. These gains are candidates to replicate, not established improvements. As a descriptive stability check, mean validation scores over the same epochs 8–13 are 0.6421, 0.6471 and 0.6507 respectively. These correlated epoch measurements are not independent statistical replicates.

`span_lr` stopped at epoch 14 even though that epoch became `best.pt`: its 0.667784 is only 0.000209 above epoch 10's 0.667575, below early stopping's `min_delta=0.002`. Checkpoint selection accepts any improvement, while stopping requires a larger one. The saved state matches that design; it is not evidence of a broken checkpoint saver.

### Source-specific results at each best checkpoint

| Run | AIHUB score | CCD score | NEXAR score | NEXAR ENTRY | NEXAR COLLISION |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.7120 | 0.6959 | 0.5637 | 4/16 | 12/16 |
| `span_lr` | 0.7173 | 0.7518 | 0.5140 | 3/16 | 11/16 |
| `span_lr_dr` | 0.7215 | 0.7479 | 0.5414 | 4/16 | 12/16 |

The newer runs improve CCD and slightly improve AIHUB. NEXAR remains weak and its total score is below baseline. In the baseline's saved per-video errors, 12 of 16 NEXAR ENTRY predictions miss tolerance: seven are early and five late. Some errors are extreme (approximately -19.6, -12.2, -9.3 seconds), while others are near misses (0.328, 0.4, 0.5 seconds). This mixture motivates separate remedies for unrelated peaks and timing precision.

All 64 NEXAR training videos exceed 512 frames, while full validation NEXAR clips contain 1,048–1,279 frames. AIHUB and CCD have only 150 and 50 frames. Cropping therefore changes long-video context selectively. However, the replay below fixes several errors without longer-context training, so crop mismatch is a remaining hypothesis, not the proven sole cause.

### Dropout and losses

By epoch 13, training scores are approximately 0.9900, 0.9891 and 0.9908. The larger dropout configuration has not prevented near-perfect fit. Evasion validation loss still rises to 1.5886 at epoch 13 in the dropout run. At the selected checkpoints it is 1.8784 / 1.5944 / 1.4022, but those checkpoints are different epochs with other hyperparameters changed.

The best evasion F1 anywhere in each observed run is 0.7196 at epoch 5, 0.7182 at epoch 10, and 0.6795 at epoch 6. It would be misleading to infer from the aggregate-selected checkpoint alone that stronger dropout uniformly improved evasion. Maintain aggregate checkpoint selection for the actual objective, and record task-specific peaks to diagnose conflicting learning dynamics.

Do not compare raw temporal cross-entropy across these runs as if the targets were unchanged: increasing Gaussian sigma changes target entropy and the achievable loss floor. Keep the current loss weights for now; measure shared-head gradient norms per task if loss balancing becomes the next question. A weak validation task with nearly perfect training accuracy is not automatically helped by increasing its coefficient.

## Evaluation inconsistencies to resolve first

### The logged runs did not configure span-limited validation

Every inspected training and exported checkpoint lacks a `validation` section. `stage2/utils/utils.py:validation_max_span_frames` returns `None` for that configuration; the trainer passes this value to validation decoding. Native `stage2/joint_test.py` also calls the decoder without a span argument.

By contrast, `outputs/submit_span_lr/inference.py` defaults to **200 frames** when the checkpoint has no configured value. Its comment saying this matches validation is inconsistent with the saved configuration and current trainer. Treat logged checkpoint scores and submission predictions as different evaluation paths until a common replay confirms parity. This assessment is based on retained code/configuration; complete executed source snapshots are not recorded for every run.

Two hundred frames is not five seconds: it means 20 s for CCD, 13.33 s for AIHUB and about 6.54–7.91 s for validation NEXAR. Since CCD/AIHUB videos are shorter than 200 frames, this restriction can only affect NEXAR in this validation set. Test FPS is explicitly unavailable in the present submission design, so do not infer it from filenames/source identities or silently assume 30 FPS.

### Decoder arithmetic changes a measurable decision

Replaying the baseline's saved float32 logits with float32 arithmetic reproduces all aggregate metrics of its logged best checkpoint (score 0.6616). Casting those logits to BF16 before decoding reproduces the saved per-video error JSON exactly, but produces score **0.6546**: two collision frame predictions differ from FP32 and one crosses the success threshold. Arithmetic precision alone explains this replay discrepancy. It does not prove the full historical pipeline's precise cause; model batching could also differ between executions.

Use explicit FP32 for decoder score arithmetic in all evaluation paths. Verify per-video equality, not just aggregate score equality, with the same frame order, detector settings and preprocessing. This does not require running the encoders in FP32.

## New experiment performed here: decoding the saved baseline logits

All following rows use the same 50 videos and fixed baseline attribute predictions. No checkpoint was retrained. These are local postprocessing measurements, not leaderboard scores.

| Decoder | ENTRY | COLLISION | Score |
|---|---:|---:|---:|
| Original pair decoder, FP32 | 50% | 80% | 0.6616 |
| Original + 200-frame maximum | 56% | 80% | 0.6826 |
| Original + actual 5-second maximum | 58% | 78% | 0.6826 |
| Tolerance probability mass, no maximum | 54% | 86% | 0.6966 |
| Tolerance probability mass + 200 frames | **60%** | **86%** | **0.7176** |
| Tolerance probability mass + 5 seconds | **60%** | **86%** | **0.7176** |
| Fixed ±3-frame probability mass + 200 frames | 54% | 78% | 0.6686 |

All 11 tested combinations, including BF16 controls and the six-second control, are retained in `decoder_replay.json`. The BF16 original / 200-frame / 5-second / 6-second scores are 0.6546 / 0.6756 / 0.6756 / 0.6756. No hidden grid search was performed.

The 200-frame FP32 decoder improves NEXAR ENTRY from 4/16 to 7/16 without changing collision accuracy. The five-second alternative ties its total but loses one NEXAR collision success while gaining one AIHUB ENTRY success. The data therefore does not show that the stricter five-second cap is superior.

### What tolerance probability mass means

The current decoder maximizes the sum of ENTRY and COLLISION logits. That selects the strongest exact pair. The score instead rewards each event separately whenever the selected frame is within 0.3 seconds of its label.

For each event and candidate frame `t`, compute `m(t) = sum p(u)` over frames whose timestamps are within ±0.3 s of `t`, where `p = softmax(logits)`. Then maximize `0.35*m_entry(e) + 0.35*m_collision(c)` over valid ordered pairs, optionally applying the span cap. A broad cluster of nearby plausible frames can thereby outweigh an isolated peak. This is a proposed decision rule aligned with the metric; the probabilities learned from Gaussian targets are not guaranteed to be calibrated event posteriors.

Compared with FP32 baseline, tolerance mass plus 200 frames corrects six ENTRY failures, introduces one new ENTRY failure, and corrects three collision failures without breaking any existing collision successes. NEXAR reaches 8/16 ENTRY and 14/16 COLLISION. These changes explain the +0.056 score gain.

**Deployment limitation:** the ±0.3-second window uses known validation FPS even when the maximum span is in frames. The 0.7176 result is not deployable as-is through an interface with no timing metadata. A fixed ±3-frame alternative does not reproduce its gain. Confirm whether legitimate test timestamps/FPS are available; otherwise keep frame-based methods explicitly separate and develop any adaptive window using training-only information. These repeated validation experiments are exploratory and require confirmation on a fresh fold.

## Next experiments, in priority order

1. **Common decoder comparison for all three best checkpoints.** Export the newer checkpoints' validation logits once. Evaluate FP32 original and FP32 200-frame decoding using exactly the same pipeline. Include tolerance-mass decoding as a diagnostic where FPS is known. The baseline already reaches 0.6826 with the frame-based decoder, above the newer models' original logged scores, but only equal-decoder comparisons can select the strongest model. Persist original frame IDs, event logits, attribute logits, labels and source per sample. Do not resume/retrain just to compare decoding.

2. **A clean dropout control.** Starting from the actual `span_lr_dr` configuration, retain LRs 5e-6/5e-6, sigmas 0.20/0.10 and all data settings, and restore dropout to 0.10 spatial, 0.10 temporal, 0.15 attribute, 0.05 LoRA. This tests the net effect of the dropout package without changing LR or target widths. If stronger dropout wins, isolate attribute-only dropout 0.30 in a later run. Do not increase all dropout rates further based on the current evidence.

3. **A matched seed replication.** Compare the chosen baseline and leading variant at seed 43, with seed 44 if the difference persists. Keep the split fixed to isolate optimization randomness. Then use source-balanced, source-group-disjoint folds to assess split sensitivity. Do not treat repeated epochs from one seed as independent trials. One additional seed of only the winning model cannot establish a paired advantage over baseline.

4. **NEXAR context experiment.** After reviewing residual errors under the improved decoder, compare frame caps 512 and 1024. If memory requires batch 1 / accumulation 4, run that batching in both control and treatment. Keep full-video validation. Record source-specific errors and actual context lengths; memory has not been benchmarked for this training setting. The objective is to reduce the observed training/evaluation context difference, not to change label definitions or infer test FPS.

5. **Pre-video ENTRY augmentation ablation.** Change only temporal mode probabilities from 70/15/15 to 85/15/0. None of the supplied manifest labels has ENTRY at frame zero, whereas this augmentation creates such targets. Inspect first-frame prediction frequency and residual early errors. Retain the mode if independent deployment data requires it and the ablation hurts those cases.

For an immediate submission with unknown FPS, first compare all checkpoints under FP32 200-frame decoding and verify the exact packaged path. For methodological progress, tolerance-aware decoding and long-video context are more strongly motivated than a larger backbone or arbitrary loss reweighting.

## Further methodological improvements

### Attribute context and confidence

The model pools attribute features using detached ENTRY/COLLISION distributions. This couples attribute quality to temporal localization while preventing attribute loss from directly adjusting those probabilities. A diagnostic evaluation can substitute a ground-truth-centered temporal window **only to measure dependence on localization**; that oracle score must not be presented as usable validation performance.

If oracle pooling substantially improves evasion, test a lightweight attribute module that reads a local temporal window around the predicted event, including surrounding scene/object tokens. Compare against current probability pooling with the backbone and data policy fixed. If oracle pooling does not help, prioritize label review and attribute regularization instead. A new independent large backbone is not justified by the present evidence.

For evasion, save logits and measure calibration/Brier score and per-class errors. Temperature scaling is an established calibration method ([Guo et al., 2017](https://arxiv.org/abs/1706.04599)), but positive temperature scaling alone preserves binary sign and therefore cannot improve macro F1 at a fixed 0.5 threshold. Threshold selection, if tried, needs training-derived out-of-fold predictions or a separate calibration split. Do not sweep many thresholds on these same 50 clips.

### Experiment bookkeeping and reliable comparison

The current W&B initialization omits loss sigmas, validation decoding, augmentation, training cap and early-stopping settings from its logged hyperparameters; checkpoints had to supply them. Log the full sanitized resolved configuration, decoder dtype/span, source/code revision, split hash and a short hypothesis for every experiment. Use immutable experiment configs. Record original and span-constrained scores under separate names when comparing decoders, and select checkpoints with the intended deployment rule.

Confirm finalists on group-disjoint folds; distinguish the original-video `source_id` group from the broad AIHUB/CCD/NEXAR domain. Keep related clips together while balancing domains across folds. The [scikit-learn cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html) describes grouped evaluation and the risk of repeatedly selecting settings on the same held-out set. The current gain estimates are not unbiased test estimates.

### Submission quality and performance

The saved profile measures a 1,279-frame NEXAR clip at 38.30 seconds end-to-end, including 20.69 seconds for RF-DETR, 6.21 seconds for RGB decoding/letterboxing and 3.07 seconds for tracking. Detector work accounts for about 54% of this measured pipeline. Disabling V-JEPA activation checkpointing saves about 0.14 seconds on the measured clip: useful but small in total. These timings are measurements of sampled clips on the profiling hardware, not full-dataset runtime guarantees.

Larger detector batches changed detection counts on multiple frames. Raw box-array differences may include reordering, so their large maxima should not be interpreted directly as matched-object displacement. Keep batch 4 until an end-to-end prediction comparison establishes acceptability. The current submission uses placeholders on load errors, per-item failures or after 40 minutes. Report actual processed/placeholder counts and effective score through that path; the check occurs between videos and is not a hard interruption of a long inference call.

If the leaderboard aggregates all stages, Stage 2's local score is insufficient to explain it: the inspected Stage 3 submission currently generates seeded random labels. Obtain per-stage results before interpreting leaderboard changes as evidence about Stage 2.

## Reproducible artifacts

- `summary.json`, `*_history.json`, `*_epochs.csv`: complete parsed history through the stated cutoff.
- `checkpoint_metadata.json`: saved settings and metrics for training and exported checkpoints.
- `decoder_replay.json`: all decoder variants, per-source scores and per-video predictions; transitions are supplied against both the saved BF16 error report and FP32 baseline.
- `comparison.png` / `comparison.pdf`: learning curves and decoder experiment comparison.
- `analyze_logs.py`, `read_checkpoints.py`, `replay_decoder.py`, `plot_comparison.py`: analysis scripts.

Only analysis artifacts were added. Training, model, config and submission files were not modified.
