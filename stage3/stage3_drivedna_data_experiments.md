Stage 3 — ADAS-Sample vs ADAS-Sample + DriveDNA-Sample

Work inside the current car-accident-competition repository and actually execute this experiment end-to-end.

Critical rule

First inspect the current local direcly thoroughly, including uncommitted/local changes, recent Stage 3 scripts/configs, preprocessing code, manifests, experiment outputs, checkpoints, W&B/logs if available, and the existing preprocessed ADAS-Sample dataset.

Determine from the actual code/data:

exactly how the current ADAS-Sample dataset was selected/filtered;

exactly how it was downloaded;

how video/CAN timestamps were synchronized;

how external videos were resampled/preprocessed;

how vEgo, direct aEgo, and steeringAngleDeg were cleaned/validated;

how clips/segments, manifests, caches and statistics were generated;

what train/validation split is currently used;

which Stage 3 method/config/checkpoint corresponds to the current best result (ADAS-only, reported real LB ≈ 0.738);

the exact training, loss, decoding and evaluation settings used by that method.

Treat the repository and existing artifacts as the source of truth.

Goal

Run one controlled data-scaling experiment:

A — current ADAS-Sample only baseline

versus

B — exactly the same method trained on ADAS-Sample + DriveDNA-Sample

The purpose is to isolate whether adding DriveDNA-Sample improves the current best Stage 3 method. Do not simultaneously redesign the architecture, loss, decoder, augmentation, thresholds, or optimization unless a minimal change is strictly necessary to support the additional dataset.

DriveDNA-Sample acquisition

Use the official Hugging Face HenryYHW/DriveDNA-Sample dataset.

Do not blindly download the entire dataset first.

First fetch/inspect the lightweight metadata, file lists and telemetry necessary to understand the dataset. Determine its actual structure and available fields from the dataset itself.

Then reproduce, as closely as technically possible, the same metadata-based filtering procedure that was previously applied to ADAS-Sample.

Translate the ADAS filtering rules to DriveDNA based on the actual DriveDNA metadata/schema rather than inventing new filtering criteria.

Produce a filtering report containing at least:

total drives/clips before filtering;

rejected count for each filtering criterion;

final accepted drives/clips;

accepted duration in hours;

distributions/ranges of vEgo, aEgo, and steeringAngleDeg;

any DriveDNA metadata incompatibility with the ADAS filtering procedure.

Only after the accepted subset has been identified, download the required video and telemetry files for those accepted samples from Hugging Face. Avoid downloading rejected video data unnecessarily.

Preprocess DriveDNA identically to ADAS

Inspect the existing ADAS preprocessing implementation and reuse/refactor it rather than creating a parallel pipeline when possible.

Convert the selected DriveDNA data into the same canonical representation used by the current preprocessed ADAS-Sample dataset.

In particular, verify rather than assume:

telemetry units;

steering sign convention;

acceleration sign convention;

timestamp origin/alignment;

video timestamps/FPS;

missing/invalid CAN values;

synchronization between video and telemetry.

Apply the same external-data temporal processing used for ADAS, including the same PTS-based synchronization/resampling behavior if that is what the current code actually uses.

The final DriveDNA examples should be indistinguishable to the Stage 3 Dataset/DataLoader from ADAS examples except for an explicit source/dataset identifier.

Run sanity checks comparing ADAS and DriveDNA after preprocessing. Check signal statistics, sample rate, duration, NaNs/gaps, acceleration distributions, steering distributions, speed distributions and several synchronized video/signal examples. Fail loudly if units, signs, timestamps or schemas appear inconsistent.

Merge

Create a merged ADAS + DriveDNA training dataset/manifests using the same canonical format.

Preserve source identity (adas, drivedna) for analysis.

Avoid train/validation leakage at the original-drive level. Do not let segments from the same original drive appear across train and validation.

Do not alter the existing ADAS-only dataset or its split destructively.

Controlled training experiment

Identify the actual current best-performing Stage 3 method from the local repository/results—the method associated with the current ADAS-only ≈0.738 leaderboard result.

Retrain that exact method on:

1.ADAS + filtered DriveDNA

Keep everything else fixed wherever possible:

architecture

initialization

losses and loss weights

decoder

thresholds

optimizer

LR schedule

augmentations

temporal crop/context

number of epochs / stopping rule

seed

evaluation implementation

The independent variable should be the addition of DriveDNA.

If the existing ADAS-only experiment is already fully reproducible and has trustworthy saved local validation results, you may use it as Experiment A instead of wasting compute retraining it, but document exactly which run/checkpoint was used. Otherwise reproduce it.

Do not tune hyperparameters specifically for DriveDNA in this experiment.

Evaluation

Evaluate both experiments with the exact same Stage 3 validation protocol currently used by the repository.

Report at minimum:

acceleration Macro-F1;

steering Macro-F1 using the competition's STOPPED-frame masking rule;

combined Stage 3 score (0.7 * accel Macro-F1 + 0.3 * steer Macro-F1);

train vs validation score;

per-class F1/confusion statistics where available.

Also report source-specific validation results if the split permits it:

ADAS validation performance;

DriveDNA validation performance;

combined validation performance.

The most important comparison is whether adding DriveDNA improves performance on the unchanged ADAS validation domain, not merely whether it scores well on DriveDNA.

Final report

Write a concise experiment report in the repository containing:

Dataset

ADAS hours/samples, DriveDNA candidates, filtering results, accepted DriveDNA hours/samples, merged dataset size, and train/val composition.

Compatibility checks

Exact mapping of DriveDNA telemetry into the existing ADAS canonical schema, including units/signs/timing and any required conversion.

Results

Training data

Accel Macro-F1

Steer Macro-F1

Stage3 score

ADAS only

...

...

...

ADAS + DriveDNA

...

...

...

Include source-specific validation metrics where possible.

Conclusion

State whether DriveDNA improved or degraded the current model and break down where the change came from (acceleration vs steering and relevant classes).

Do not claim improvement from training loss alone.

Engineering requirements

Make the DriveDNA acquisition/filter/preprocessing reproducible and resumable. Avoid redownloading files that already exist. Keep raw downloads, filtered metadata, processed data, manifests and experiment outputs clearly separated.

Before starting expensive preprocessing or training, run small smoke tests through the complete pipeline and verify that DriveDNA and ADAS samples produce identical tensor/schema contracts.

Commit/code the reusable DriveDNA support and experiment configuration, but do not unnecessarily rewrite the working Stage 3 pipeline.

Most importantly: inspect the existing implementation first, reproduce what was actually done for ADAS, and make the smallest changes necessary for a clean ADAS-only vs ADAS+DriveDNA experiment.