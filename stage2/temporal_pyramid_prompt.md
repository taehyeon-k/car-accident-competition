Before modifying anything, inspect the current Stage 2 implementation, experiment code, geometry-DINO feature pipeline, manifests, configs, training history, and reports. Do not blindly follow this prompt if the repository structure or implementation details differ. Preserve the existing working baseline and submission code.

1. Current problem

Stage 2 predicts:

entry_frame
collision_frame
entry_side
evasion_space

The current strong approach uses a frozen geometry-tuned DINOv3 ViT-S representation and a relatively small temporal head.

Current major issue:

train score can approach ~0.99
validation is around ~0.65
real leaderboard is around ~0.42

There is also a strong apparent relationship between video frame count and performance.

Examples observed:

MM-AU videos: roughly ~100 frames on average, validation score around ~0.8
Nexar videos: often ~1000 frames, validation score around ~0.5

Do NOT assume this proves frame count is the only causal factor because source/domain also differs. Part of this task is to test whether performance genuinely degrades with frame count.

The Stage 2 inference model must remain:

FPS-blind
duration-blind
timestamp-blind

It may use:

ordered frames
original frame numbers
total available frame count for deterministic sampling only
normalized relative frame positions

It must NOT use FPS, seconds, inferred FPS, or assumed physical duration as model inputs.

Final predictions must map back to original frame numbers.

2. Hypothesis to test

The current fixed temporal sampling strategy likely creates an unfair problem across sequence lengths.

For example:

100-frame video → 128 requested samples means almost every frame is visible
1000-frame video → 128 samples means roughly one observed frame every ~8 original frames

Therefore long videos lose much more temporal information.

The experiment should test whether a length-adaptive temporal pyramid improves performance, especially on long videos, without causing overfitting.

Do NOT immediately build a large ActionFormer/TriDet/T-DEED clone.

The purpose is a compact controlled experiment.

3. Main temporal pyramid design

Use the SAME frozen geometry-tuned DINOv3 ViT-S features as the current best Stage 2 experiments.

Do not fine-tune the backbone for this experiment.

The architecture should be approximately:

geometry-DINO features
        ↓
small shared frame/spatial encoder
        ↓
L0: finest temporal sequence
        ↓ stride-2
L1
        ↓ stride-2
L2
        ↓ stride-2
L3
        ↓ optional L4 for very long sequences

then top-down:

L3
 ↓ upsample
+ L2
 ↓
L2'
 ↓ upsample
+ L1
 ↓
L1'
 ↓ upsample
+ L0
 ↓
L0'

L0'
 ├── ENTRY dense logits
 └── COLLISION dense logits

This is conceptually a 1D temporal FPN/U-Net.

Important:

No hard coarse prediction should determine which region survives.
Coarse levels provide context only.
Final ENTRY/COLLISION prediction must come from the highest-resolution fused sequence.
Therefore there should be no classical coarse-to-fine error accumulation.

Keep the architecture lightweight.

Suggested starting values:

hidden dimension: 128
3 or 4 pyramid levels
stride-2 temporal downsampling
small residual depthwise Conv1D blocks
kernel size 3 or 5
dropout around 0.3–0.4
optional one local-attention block only at the lowest-resolution level
avoid global self-attention over the full high-resolution sequence

Target trainable parameter count:
preferably <500k if possible, and definitely not substantially larger than the current ~1M temporal head.

4. Length-adaptive finest resolution

The key experiment is that L0 must NOT always have 128 positions.

Create a configurable sampling rule based only on original frame count.

Test a compact rule such as:

N <= 128:
    use approximately all available frames

128 < N <= 256:
    K = 128

256 < N <= 512:
    K = 192

512 < N <= 768:
    K = 256

N > 768:
    K = 320 or 384

You may replace this with a cleaner equivalent formula if appropriate, but keep the experiment easy to interpret.

Frame count is allowed to control deterministic sampling density.

Do NOT provide raw num_frames as an explicit learned feature to the classifier.

The model must not learn:

100 frames → MM-AU
or
1000 frames → Nexar.

The purpose of frame count is only to determine how densely the video is sampled.

All sampled positions must retain their original frame-number mapping.

5. Experiments

Run only the following essential experiments first.

B0 — current fixed-128 baseline

Reproduce the strongest current comparable Stage 2 model with:

same frozen geometry-DINO checkpoint
same train/val split
same seed
fixed 128 temporal samples
same or closest current temporal head

This is the control.

P1 — temporal pyramid, fixed 128

Use the new temporal pyramid architecture but still force every video to 128 positions.

Purpose:

Determine whether gains come from the pyramid itself.

P2 — temporal pyramid + length-adaptive sampling

Use the same pyramid as P1 but apply the length-adaptive K(N) sampling rule.

Purpose:

Determine whether giving long videos more fine temporal resolution solves the frame-count degradation.

This is the most important experiment.

P3 — P2 + temporal sampling augmentation

Keep P2 architecture.

During training randomly vary the resolution around the default K(N).

For example, if the default is 256, randomly choose a nearby supported resolution such as:

192
256
320

Also allow mild:

sampling-grid jitter
frame dropping
monotonic temporal warp

Do not use FPS.

Purpose:

Prevent the network from depending on one exact sampling density.

P4 — P3 + cross-view temporal consistency

For each training video, create two independently sampled temporal views.

Example:

View A:
K=192, one sampling grid

View B:
K=320, different grid + mild drop/warp

Run both through the same network.

Map ENTRY and COLLISION distributions back to a shared normalized/original-frame coordinate system.

Add a small consistency loss encouraging both views to identify the same event.

Prefer a distance-aware loss such as CDF-L1 / 1D Wasserstein-like consistency rather than only Jensen-Shannon between discrete bins.

Keep this loss small.

Purpose:

Explicitly train sampling-density invariance.

Do not add unrelated architectural changes in P4.

6. Do NOT add exact-frame refinement yet

Do not implement a separate fine refinement model in this experiment.

First determine whether the temporal pyramid and length-adaptive high-resolution sequence solve the long-video problem.

If P2/P3/P4 clearly improve long-video performance, exact-frame refinement can be evaluated afterward as a separate experiment.

Do not confound the result now.

7. Event heads

Use dense per-position ENTRY and COLLISION prediction at L0'.

Keep the loss comparable to the best current setup.

Do not perform a large loss sweep.

If the existing exact-anchor or normalized-soft target is the strongest current setting, use that consistently across B0/P1/P2/P3/P4.

Always enforce:

entry_frame <= collision_frame

during decoding.

Do not introduce second-based loss widths.

8. Optional weak pyramid supervision

If training is unstable, allow weak auxiliary ENTRY/COLLISION supervision at intermediate pyramid levels.

For example:

L0 final loss weight = 1.0
L1 auxiliary = 0.25
L2 auxiliary = 0.10
L3 auxiliary = 0.05

Do not let coarse-level losses dominate.

Final performance must be driven by the highest-resolution output.

9. Frame-count analysis — mandatory

This experiment is not useful unless performance is explicitly analyzed against frame count.

Create frame-count bins, for example:

<150
150–300
300–500
500–750
750–1000
>1000

For every experiment report:

number of videos in each bin
ENTRY ±0.3s
COLLISION ±0.3s
entry_side Macro-F1
evasion_space Macro-F1
overall competition-style score
normalized ENTRY error
normalized COLLISION error

Also report continuous correlations between:

num_frames
ENTRY error
COLLISION error
total score

Do this globally and, where enough samples exist, separately by source.

At minimum report separate curves for:

MM-AU
Nexar
AIHub
CCD
CausalCrash if available

This is critical because we need to distinguish:

frame-count failure

from:

dataset/source domain shift.

10. Most important comparison

Create a table similar to:

Model	Sampling	<150	150–300	300–500	500–750	750–1000	>1000	Overall
B0	fixed 128	
	
	
	
	
	
	

P1	pyramid fixed 128	
	
	
	
	
	
	

P2	pyramid adaptive	
	
	
	
	
	
	

P3	adaptive + augmentation	
	
	
	
	
	
	

P4	adaptive + consistency	
	
	
	
	
	
	


The central question is NOT merely which method has the highest global validation score.

The key questions are:

Does score decline as frame count increases?
Does that decline exist within the same source?
Does P1 help over B0?
Does P2 specifically improve long videos?
Does P2 preserve performance on short videos?
Does P3 improve robustness across different sampling densities?
Does P4 further reduce dependence on sample density?
Which method produces the flattest performance-vs-frame-count curve?
Does improved frame-count robustness correlate with better source generalization?
Does the temporal pyramid overfit less than the existing model?
11. Overfitting diagnostics

For every run save:

train loss
validation score per epoch
ENTRY/COLLISION metrics per epoch
side/evasion F1
parameter count
best epoch
gap between best validation epoch and final epoch

Check whether:

train performance continues increasing
while validation starts decreasing

as happened in prior experiments.

Use early stopping based on validation metrics.

Do not compare models using final epoch weights.

Also report trainable parameter counts.

Prefer architecture changes that improve long-video performance without increasing model capacity substantially.

12. Sampling robustness evaluation

For the best two models, evaluate the same validation video under multiple artificial sampling densities.

For example:

96
128
192
256
320

when supported by sequence length.

Map predictions back to original frame coordinates.

Measure:

normalized prediction standard deviation across views
original-frame prediction range
score variation across sample counts

A robust model should produce similar ENTRY/COLLISION predictions under different sampling densities.

13. Resource constraints

The competition has limited inference time, so benchmark:

encoded frames/video
temporal-head runtime
peak VRAM
total parameter count

Because DINO is frozen and likely the dominant cost, specifically report how adaptive sampling increases DINO frame count compared with fixed-128.

If cached features are available for experiments, use them.

Do not claim runtime viability from cached-feature training alone; estimate or benchmark raw-frame inference separately if possible.

14. Implementation principles

Keep implementation modular.

Suggested structure:

stage2/temporal_pyramid_experiments/
    models.py
    sampling.py
    data.py
    objective.py
    train.py
    robustness.py
    analyze_frame_count.py
    configs/
    results/

Adapt to repository conventions if a better existing experiment framework already exists.

Do not duplicate utilities unnecessarily.

Do not break the current Stage 2 submission implementation.

15. Execution order

Actually run the experiments rather than only writing code.

Priority:

B0
→ P1
→ P2
→ P3
→ P4

If compute is limited, B0/P1/P2 are mandatory.

P2 is the main experiment.

Do not add extra architecture variants until these results are understood.

16. Final report

Save a report such as:

reports/stage2_temporal_pyramid_framecount_experiments.md

Include:

exact architecture
sampling rules
parameter count
training protocol
overall metrics
metrics by frame-count bin
metrics by source
frame-count vs error plots
sampling-robustness results
train-vs-validation curves
runtime
failure cases

End the report with explicit answers:

Is frame count actually an independent performance problem?
How much of MM-AU vs Nexar performance difference remains after controlling for frame count?
Does a temporal pyramid help even under fixed-128 sampling?
Does length-adaptive sampling improve long-video ENTRY?
Does it improve long-video COLLISION?
Does adaptive sampling hurt short videos?
Does temporal augmentation reduce dependence on input density?
Does cross-view consistency add meaningful robustness?
Which configuration has the smallest validation degradation as frame count increases?
Is the improvement large and stable enough to justify replacing the current Stage 2 temporal head?

Do not select a model merely because it has the highest single validation score.

The preferred result is a model with:

competitive overall validation
substantially better >500 / >750 / >1000-frame performance
flatter score-vs-frame-count curve
lower temporal-sampling sensitivity
smaller train/validation gap
reasonable inference cost
strictly FPS-blind inference

At completion print:

best experiment ID
overall score
scores by frame-count bin
ENTRY/COLLISION metrics
trainable parameter count
runtime
best checkpoint path
report path
whether the temporal pyramid hypothesis was supported or rejected
recommended next experiment