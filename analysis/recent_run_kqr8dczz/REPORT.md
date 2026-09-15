# Recent training run: kqr8dczz

## Scope and evidence

Analyzed the local W&B binary log from `run-20260914_164354-kqr8dczz`, its saved configuration and summary, current manifests, and restricted metadata reads from both checkpoints. Extracted 114 history records, including all 19 epoch summaries. Binary record checksums passed; the best and final checkpoint metrics agree with the log. No training or new inference was performed. No per-video predictions were present in these artifacts, so error direction, annotation quality, calibration and individual failure cases remain unmeasured.

The older `analysis/joint_run_history.json` is a different run. Its 0.6819 peak is not the recent run's result and is not a controlled comparison.

## Result

- Started September 14, 2026 at 16:43 UTC; approximately 3.70 hours on one L40S.
- Best checkpoint: epoch 13, optimizer step 663, score **0.6616215**. Stored epoch 12 is zero-based.
- Final checkpoint: epoch 19, step 969, score **0.6418820**. Early stopping triggered after six validations without the configured improvement.
- Lowest validation loss: **3.2409 at epoch 4**, versus 3.4819 at the best-score epoch.
- Baseline: 256-D head; one spatial and one temporal block; rank-8 LoRA, DINO final two blocks, V-JEPA final four; learning rates 2e-5 / 1e-5 / 3e-4 for DINO / V-JEPA / head; batch 2, accumulation 2; BF16; 512-frame training cap; temporal modes 70/15/15; event target sigma 0.10 seconds for both events; attribute dropout 0.15.

| Metric | Train, epoch 13 | Validation, epoch 13 | Validation, epoch 19 |
|---|---:|---:|---:|
| Competition score | 0.9900 | 0.6616 | 0.6419 |
| ENTRY within 0.3 s | 99.5% | 50.0% | 42.0% |
| COLLISION within 0.3 s | 100.0% | 80.0% | 78.0% |
| Entry-side macro F1 | 0.9652 | 0.8182 | 0.8193 |
| Evasion-space macro F1 | 0.9801 | 0.5593 | 0.6599 |

Training metrics are collected during optimization on augmented, often cropped samples. Validation uses evaluation mode and complete videos. The gap strongly suggests a generalization problem, but its magnitude also reflects different evaluation conditions. Temporal cross entropy depends on sequence length, so a raw train/validation event-loss gap alone is not proof of overfitting.

## Main findings

### 1. NEXAR ENTRY is the clearest source-specific bottleneck

At the best overall checkpoint:

| Source | Validation clips | ENTRY accuracy | COLLISION accuracy | Score |
|---|---:|---:|---:|---:|
| AIHUB | 18 | 66.7% (12/18) | 83.3% | 0.7120 |
| CCD | 16 | 56.3% (9/16) | 81.3% | 0.6959 |
| NEXAR | 16 | 25.0% (4/16) | 75.0% | 0.5637 |

NEXAR ENTRY was already 18.75% at epoch 4, remained 18.75% at epoch 10, and ended there. This is not only a late-training decline.

All 64 NEXAR training clips exceed 512 frames; their median length is 1,210. All 16 validation NEXAR clips are 1,048–1,279 frames, with median 1,202.5. AIHUB clips are 150 frames and CCD clips 50. Consequently, the cap selectively changes NEXAR's training context. At epoch 13, mean original training length was 445.5 frames and mean actual training length 204.9; the cap affected 24.9% of samples. Longer search space and changed relative event positions are plausible contributors, not established causes.

The sources also differ in frame rate (CCD 10, AIHUB 15, NEXAR about 30). The index-based temporal receptive fields therefore cover different durations. Keep this as a secondary hypothesis; the model intentionally uses frame-index sampling.

### 2. Evasion deteriorates while the overall score improves

Evasion validation F1 peaks at 0.7196 in epoch 5, then is 0.5593 at epoch 13. Its validation loss increases from 0.6719 to 1.8784 over the same interval, while training evasion F1 reaches 0.9801. This is consistent with increasingly confident mistakes and overfitting, but probabilities are needed to measure calibration directly. Best overall checkpoint selection legitimately trades off the four tasks; it does not select the best checkpoint for each task.

### 3. More epochs or a larger model are low-priority experiments

Validation reaches 0.6341 at epoch 4 and fluctuates around a plateau as training approaches perfection. Late training brings modest event improvements but attribute degradation. The present evidence favors controlling adaptation and investigating ENTRY errors before increasing capacity.

### 4. Class/source imbalance is not the obvious explanation

Training source counts are 70 AIHUB / 67 CCD / 64 NEXAR. Evasion labels are reasonably balanced within each source. There is no overlap in manifest `source_id` between training and validation, although that does not establish absence of visual duplicates. Weighted sampling or class-weighted losses are not the first change to try.

## Recommended next experiments

Run each against the original baseline first, retaining the same manifests, seed, head size, 20-epoch schedule and checkpoint criterion. Use a new output directory per experiment. Do not resume baseline optimizer state for a changed experiment.

### First: inexpensive diagnostic evaluation

Evaluate `best.pt` with per-video outputs: signed and absolute ENTRY/COLLISION error in seconds, all four target/prediction pairs, event distributions, attribute probabilities, source, length, and normalized event position. Inspect all 16 NEXAR validation clips, including ENTRY ground truth and predictions, plus evasion mistakes. Determine whether ENTRY is slightly mistimed, locks onto collision, or selects an unrelated early frame. Evaluate the training manifest in evaluation mode on full unaugmented videos, too. This gives a comparable train/validation gap without starting a new training run.

A fixed-position baseline estimated on training data only would also help detect how much event timing can be explained by dataset structure. Any diagnostic crop selected from ground-truth labels is an oracle analysis and must never be reported as submission performance.

### Experiment 1: reduce backbone adaptation

Set `dino_lora_lr: 4.0e-6` and `vjepa_lora_lr: 2.0e-6` (both fivefold lower); keep `new_lr: 3.0e-4`. This tests one hypothesis: excessive adaptation of pretrained features on 201 videos. Look for improved full-video validation score and evasion performance without losing collision accuracy. These values are proposed trial settings, not proven optima.

If promising, separate which branch needs adaptation with DINO-only and V-JEPA-only ablations. The present code always attaches LoRA and rejects zero blocks. A true frozen branch needs an explicit freeze/optimizer implementation; merely setting its learning rate to zero does not deliver the memory savings of freezing autograd.

### Experiment 2: reduce synthetic first-frame ENTRY supervision

Change temporal probabilities to `full_video: 0.85`, `ordinary_crop: 0.15`, `pre_video_entry: 0.0`. This removes only synthetic pre-video ENTRY examples, reallocating their probability to full video. None of the 251 current manifests has ENTRY at frame zero, while this mode deliberately creates that target. It may teach a boundary shortcut, particularly on cropped long videos. Compare NEXAR ENTRY, signed error and first-frame prediction frequency. The hypothesis is unconfirmed; retain the mode if the true deployment/test distribution contains pre-video ENTRY cases and the ablation hurts those cases.

### Experiment 3: increase temporal context for long videos

Compare the 512-frame cap with 1,024 frames. To fit memory, use batch size 1 and accumulation 4 for both the 512 control and the 1,024 experiment, keeping nominal effective batch size 4. Measure memory first; available GPU headroom has not been established here. The 1,024 cap reduces but does not eliminate the NEXAR mismatch. If the result improves, test full clips with a suitable memory strategy. Keep validation on full videos throughout. Do not change temporal sampling stride in this experiment.

This experiment has the most direct connection to the NEXAR failure, but costs more and needs a matched batching control; run it after inspecting per-video errors.

### Follow-up experiments

- **Attribute regularization:** `attribute_dropout: 0.30` versus 0.15, all else fixed. This affects both attribute heads; check that entry-side F1 survives. If evasion alone needs stronger regularization, add a separate evasion dropout setting in a later experiment.
- **ENTRY target width:** `entry_sigma_seconds: 0.15` versus 0.10, leaving collision sigma at 0.10. Prioritize only if errors are near the 0.3-second tolerance or labels appear temporally ambiguous. It will not necessarily fix completely wrong event selection.
- **Frozen-feature baseline:** same current head/data/augmentations, both backbones truly frozen. The older run is not a substitute for this controlled comparison.

## How to decide

The score is 0.35 ENTRY + 0.35 COLLISION + 0.15 side F1 + 0.15 evasion F1. Fixing one validation video's ENTRY changes the total score by **0.007**, holding other metrics fixed. Improving NEXAR ENTRY from 4/16 to 8/16 would add **0.028**. Restoring evasion F1 from 0.5593 to 0.7196 would add about **0.024**. These are sensitivity calculations, not achievable combined predictions.

Use 0.02 absolute score improvement as a screening signal, not statistical proof. Log all components and per-source results, compare stability across nearby epochs, and confirm the leading candidate against baseline with two additional matched seeds (for example 43 and 44) on the same split. With only 50 validation videos, small gains may be a few changed examples. Estimate paired uncertainty from per-video predictions when available; aggregate logs cannot support a proper paired bootstrap.

Then confirm finalists with source-balanced, source-group-disjoint folds. Do not continually tune against the same 50 clips. See the [scikit-learn cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html) for group-aware splitting and the distinction between model selection and final evaluation. The local score's equivalence with the external competition evaluator was not verified in this analysis.

## Artifacts

- `history.json`: all extracted W&B history records.
- `epochs.csv`: 19 complete epoch metric rows, including source slices.
- `learning_curves.png` and `learning_curves.pdf`: four-panel analysis figure.
- `extract_history.py`: read-only W&B log extraction with checksum checks (requires W&B protobuf definitions; used W&B 0.30.0).
- `plot_history.py`: reproducible matplotlib figure and CSV export.

Extraction follows the file format documented in the [official W&B datastore source](https://github.com/wandb/wandb/blob/v0.19.11/wandb/sdk/internal/datastore.py). The analysis uses the local log; it does not contact W&B or create a tracking run.
