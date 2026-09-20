The objective is to improve Stage 2 of the DACON dashcam intentional-accident competition. (https://dacon.io/competitions/official/236753/overview/description)
Do not start by redesigning the project. First inspect the repository, current Stage 2 code, current geometry-DINO checkpoints, cached features, manifests, training reports, submission runtime, and available GPU/storage. Preserve the currently working submission code and checkpoints.
1. Background
A geometry-adapted DINOv3 ViT-S backbone has already been trained for driving geometry. (inside /workpsace/geometry_pretrain)
The geometry adaptation included road/lane geometry, object/contact/depth information and temporal correspondence/flow. Existing experiments showed that the no-anchor geometry-adapted DINO representation transfers better to Stage 2 than the original DINO representation.
A very simple downstream Stage-2 head using this geometry-aware representation achieved approximately:
real leaderboard Stage-2 score ≈ 0.41
A previous much more complicated joint architecture achieved approximately:
real leaderboard Stage-2 score ≈ 0.44
Therefore the purpose of this work is NOT to immediately recreate a large detector/tracker/depth/flow/joint model.
The working hypothesis is:
the geometry-DINO representation is already strong and most remaining performance is limited by temporal event localization, task-specific pooling, sampling invariance and precise readout.
The goal is to experimentally test that hypothesis.
2. Competition targets
Stage 2 predicts exactly four outputs:
collision_frame
entry_frame
evasion_space
entry_side
Internal Stage-2 weighting is:
collision_frame: 0.35
entry_frame: 0.35
evasion_space: 0.15
entry_side: 0.15
ENTRY and COLLISION are point events.
ENTRY means the first frame in which the victim vehicle's wheel touches the ego vehicle's lane.
COLLISION means the actual first contact between the ego/dashcam vehicle and victim vehicle.
entry_side is binary LEFT/RIGHT based on the image coordinate system.
evasion_space is binary 0/1 and represents whether physically usable avoidance/drivable space exists at collision.
Private Stage-2 examples are direct ego-vehicle/victim-vehicle collisions.
3. Critical inference constraint: FPS-blind
The private Stage-2 input is a directory of extracted frames.
Frame filenames contain original frame numbers.
Do NOT assume access to:
FPS
duration_seconds
timestamps
frame-to-time mapping
The evaluator has hidden frame-time mapping, but the submitted model does not.
Therefore all new candidate models must support a STRICT FPS-BLIND mode.
During inference, allowed temporal information is only:
    • ordered frames;
    • original frame numbers parsed from filenames;
    • number of available frames;
    • relative/normalized sequence position;
    • relative differences between observed original frame indices.
The model must return valid original frame numbers.
Do not infer FPS from assumed dataset conventions.
Do not hard-code 10, 15, 20, 25 or 30 FPS.
4. First audit the current baseline
Inspect at least:
stage2/geometry_pretrain/downstream_probe.py
stage2/utils/joint_losses.py
stage2/utils/joint_metrics.py
submission_tools/train_stage2_probe.py
submission_tools/train_fps_stage2.py
submission_tools/fps_stage2/runtime.py
submission_tools/fps_stage2/sampling.py
reports/stage2_geometry_dino_pretraining.md
Document the exact current pipeline.
Pay particular attention to the current train/inference temporal sampling mismatch.
Current training code uses native_fps when constructing approximately 10-Hz cached positions and also creates frame_seconds, entry_s and collision_s.
Current private-style runtime instead selects a fixed number of normalized locations directly from the available frame numbers without FPS.
Quantify how much this matters.
Implement a strict fps_blind=True training mode in which native_fps and duration are NEVER accessed by:
    • feature sampling;
    • model inputs;
    • target construction;
    • loss functions;
    • decoding;
    • temporal augmentation.
It is acceptable for an offline validation evaluator to use training-set FPS only AFTER predictions have been produced, solely to calculate the DACON-style ±0.3 s metric.
The model itself and its model-selection inputs must remain deployable without FPS.
Also create frame-index / normalized-position metrics that require no FPS.
5. Do not destroy the existing baseline
Create a clean experimental module, conceptually:
stage2/spotting_experiments/
Adapt this name if the repository conventions suggest something better.
Existing Stage 2 production code must continue to run.
Every experiment must be configurable through YAML/JSON/CLI rather than by editing Python constants.
Every experiment must use the same fixed folds and seeds where possible.
6. Backbone policy
For the initial architecture screening, use the SAME geometry-tuned backbone/checkpoint that produced the current strong simple-head result.
Find the actual checkpoint on disk instead of guessing the path.
Support:
--backbone-checkpoint PATH
and support both DINOv3 ViT-S and ViT-L dimensions cleanly.
If a geometry-tuned ViT-L checkpoint is available, identify it and benchmark it.
If no geometry-tuned ViT-L checkpoint exists, DO NOT substitute vanilla ViT-L and call that a fair ViT-S-vs-L comparison.
First screen temporal heads cheaply with the proven geometry-tuned ViT-S representation, then make the winning design ViT-L-compatible.
Keep the backbone frozen during the initial head comparison.
Only test partial backbone fine-tuning after selecting the strongest head.
7. Feature extraction
The existing probe reduces DINO patch tokens to a 7×10 grid.
Preserve that configuration as the baseline.
Implement caching that can support at least:
7x10
14x20
without rerunning every head experiment.
If practical, support selected final-layer feature aggregation:
last layer only
mean of final 4 layers
learned weighted final 4 layers
Do not run all combinations initially. Feature extraction experiments come after the temporal-head screening unless caching makes them effectively free.
8. Experiment E0 — exact baseline
Reproduce the existing FPS-independent submission pipeline as faithfully as possible.
Record:
    • train/validation score;
    • ENTRY accuracy;
    • COLLISION accuracy;
    • entry-side Macro-F1;
    • evasion-space Macro-F1;
    • parameter count;
    • VRAM;
    • inference speed;
    • temporal sampling behavior.
Then reproduce it in strict FPS-blind training mode.
This gives:
E0-current
E0-fpsblind
Do not continue until these reproduce reliably.
9. Experiment E1 — T-DEED-lite precise spotting
Implement a SMALL precise-event-spotting head inspired by T-DEED.
Do NOT port the complete T-DEED video architecture.
Input:
per-frame geometry-DINO features
Output for each temporal location and each of the two events:
event classification logit
event displacement / offset
The classification head determines whether ENTRY/COLLISION occurs at or near a sampled position.
The displacement head predicts the exact event location relative to that temporal anchor.
Do all localization in original-frame or normalized-sequence coordinates, never seconds.
For a sampled position corresponding to original frame f_k, predict an offset which can be transformed back into an original frame number.
Test bounded normalized offsets rather than unrestricted regression.
Use high temporal output resolution.
Implement a compact multiscale encoder-decoder:
input sequence
→ local temporal block
→ downsample temporal context
→ deeper local temporal block
→ upsample
→ skip connection
→ full-resolution event tokens
Keep it small enough for 251 labeled videos.
Suggested initial capacity:
hidden = 192 or 256
2–3 temporal scales
2 blocks per scale maximum
dropout ≈ 0.2–0.4
Do not blindly copy sports-specific settings.
The most important ablation is:
classification only
vs
classification + displacement
10. Experiment E2 — TemporalMaxer-lite
Implement the simplest serious candidate.
The hypothesis is that the geometry-DINO features already contain most of the required information.
Use local temporal max-pooling at several receptive fields, for example conceptually:
3
5
9
17
or an equivalent multiscale design.
Fuse local pooled features with the original feature through residual projection.
Then predict dense ENTRY/COLLISION logits.
No global self-attention.
No interval proposal generation.
No TAL-style NMS.
No action-duration regression.
Keep parameter count substantially below E1.
This experiment is extremely important and must not be skipped.
11. Experiment E3 — ASFormer-lite/local temporal attention
Implement a small local-attention temporal model.
Use local/windowed/dilated temporal attention rather than unrestricted full-sequence attention.
Candidate initial sizes:
2 layers
4 layers
with hidden width approximately:
192–384
Avoid a large Transformer.
The purpose is to determine whether learnable local temporal relationships improve over TemporalMaxer.
Compare fairly under the same sampling and loss.
12. Point-event supervision experiments
Existing loss uses Gaussian targets parameterized in seconds.
Create FPS-blind alternatives.
At minimum test:
A. exact-anchor cross entropy;
B. soft target in sampled-index distance;
C. soft target in normalized clip distance;
D. classification-radius + displacement, as in precise event spotting.
Do not assume a ±0.3-second radius can be converted into a fixed frame radius at private inference.
For normalized targets, sweep only a small number of sensible widths.
For classification+displacement, positives can be anchors near the event in normalized/index space and displacement handles the exact location.
Preserve the physical constraint:
entry_frame <= collision_frame
at decoding.
Do not impose an arbitrary maximum time gap in seconds.
If a maximum-span prior is tested, express it as a normalized clip fraction and report it as a separate ablation.
13. Temporal resampling augmentation
Unknown private FPS makes temporal-scale robustness important.
Implement training-time cadence augmentation without FPS.
For each training video, randomly vary:
    • number of normalized samples;
    • normalized-grid offset/jitter;
    • local sample density;
    • optional frame dropping.
Example candidate sample counts:
96
128
160
Do not encode the chosen sampling count as a proxy for FPS.
Map all predictions back to original frame indices.
Test whether models trained with randomized temporal sampling are more stable under simulated inference sampling changes.
Add a robustness evaluation:
run each validation video through multiple artificial temporal subsamplings and measure prediction variance in normalized/original-frame coordinates.
14. Experiment E4 — event-conditioned categorical heads
The current probe globally pools the temporal sequence before predicting:
entry_side
evasion_space
Replace this with event-aware task heads.
entry_side
This task should use geometry near ENTRY.
Use the predicted ENTRY probability distribution to perform differentiable soft temporal pooling around entry.
Construct separate contextual summaries such as:
before-entry
at-entry
after-entry
Do not require a hard argmax during training.
Because direction is inherently spatial, preserve spatial patch information.
Test a lightweight spatial directional representation such as:
left-half pooled geometry
center pooled geometry
right-half pooled geometry
around the predicted entry event.
Then classify LEFT/RIGHT.
evasion_space
Use features around and immediately preceding COLLISION.
Construct:
pre-collision context
collision-centered context
and preserve spatial patch layout.
Use a lightweight spatial-attention pooling head over the geometry-DINO patch grid.
This task should not depend primarily on a single globally pooled video token.
Training stability
Initially allow event-conditioned heads to use ground-truth event distributions during an early warmup.
Then transition to predicted soft event distributions.
Also test fully differentiable predicted-distribution pooling from the beginning.
Report which works better.
15. Experiment E5 — temporal multi-view inference
Implement an inexpensive temporal test-time ensemble.
Run the same trained model on several normalized sampling grids, for example:
96
128
160
and/or shifted/jittered versions of the 128 grid.
Map logits and offset predictions back into original-frame coordinates.
Fuse predictions by interpolation/weighted averaging in original-frame coordinate space.
For categorical predictions, average probabilities.
This must remain within the DACON inference budget.
Measure the improvement per additional DINO forward cost.
16. Experiment E6 — coarse-to-fine event refinement
Implement only after E1–E5 are working.
Pass 1:
uniform normalized sampling over the whole video.
Output coarse:
ENTRY
COLLISION
Pass 2:
around each coarse event, load consecutive ORIGINAL frames from a local neighborhood and run dense refinement.
Because private FPS is unknown, neighborhood sizes must not be specified in seconds.
Test normalized-video radii, e.g. approximately:
±2%
±4%
±8%
of available frames, with sensible caps.
The local refiner should predict the exact original frame.
ENTRY and COLLISION may use separate refinement heads if validation supports that choice.
Preserve:
entry <= collision
17. Experiment E7 — AdaSpot-like spatial refinement
This is optional and should only run after the temporal experiments.
Current baseline spatial pooling is only 7×10.
Test whether precise events benefit from more local geometry.
First compare:
7x10
vs
14x20
for the best temporal head.
If 14×20 is clearly beneficial, implement a lightweight coarse-to-fine spatial strategy.
Use low-resolution/full-frame DINO geometry for global context.
Near candidate ENTRY/COLLISION frames, selectively retain/process higher-resolution patch information.
Do NOT introduce an object detector solely for this experiment.
Do NOT run expensive high-resolution processing across every frame unless runtime benchmarking proves it feasible.
The hypothesis is that:
wheel-to-lane contact
and
vehicle-to-vehicle contact
may require spatial detail that 7×10 pooling discards.
18. Optional DINO layer experiment
For the winning temporal architecture, compare:
final layer
mean final 4 layers
learned weighted final 4 layers
Do not perform this sweep independently for every temporal head.
19. Do NOT prioritize these approaches
Unless experiments strongly justify them, do not reintroduce:
    • RF-DETR tracking pipelines;
    • explicit multi-object tracking;
    • Depth Anything as an online branch;
    • optical-flow inference as a separate online branch;
    • giant global Transformers;
    • full ActionFormer/TriDet proposal machinery;
    • interval regression/NMS;
    • a VideoMAE replacement backbone;
    • accident-anticipation models trained to predict risk seconds before collision.
The geometry-tuned DINO already encodes geometry and correspondence information.
The experiment should determine how effectively that representation can be read out.
20. Evaluation protocol
There are only approximately 251 fully labeled videos.
Avoid choosing models from one noisy split.
Use the existing source/label-stratified folds where appropriate.
Experiment protocol:
Phase A — cheap screen
Use the existing fixed train/validation split.
One fixed seed.
Frozen geometry-DINO cached features.
Compare:
E0-fpsblind
E1
E2
E3
Only architecture/loss changes needed for screening.
Phase B — robust comparison
Take the best two candidates plus E0.
Run:
5-fold CV × 3 seeds
using exactly the same folds and seeds.
Report mean ± standard deviation.
Also compute paired per-fold/per-seed score differences relative to E0.
Do not select a winner from a single lucky run.
Phase C — task-head improvement
On the best temporal localizer, evaluate E4.
Measure all four target metrics independently.
Phase D — inference improvements
Evaluate E5 and E6 without retraining where possible.
Phase E — representation/spatial improvement
Only now test E7 and final-4-layer feature aggregation.
21. Metrics
Always report separately:
ENTRY ±0.3s accuracy — evaluator only
COLLISION ±0.3s accuracy — evaluator only
entry_side Macro-F1
evasion_space Macro-F1
competition-style weighted score
Also add FPS-independent temporal metrics:
absolute original-frame error
normalized-frame error = abs(pred-gt)/(N-1)
median normalized error
P(error <= 0.5%, 1%, 2%, 5% clip length)
The model and loss must not consume validation FPS.
An evaluator may use known validation FPS after prediction solely to estimate official ±0.3-second correctness.
22. Domain-generalization diagnostics
Because leaderboard performance appears substantially below internal validation for the simple-head family, investigate generalization explicitly.
For every final candidate, evaluate robustness to:
    • temporal subsampling;
    • duplicated frames;
    • dropped frames;
    • changed normalized sample count;
    • different source datasets;
    • clip-length distribution;
    • frame-resolution/aspect-ratio differences.
Break validation results down by source.
Check whether performance depends suspiciously on training-source FPS or clip duration even though neither is available privately.
Plot score versus:
num_frames
source
training native FPS
for diagnostic purposes only.
A good private model should not collapse when temporal cadence changes.
23. ViT-L transfer
After selecting the best architecture on the geometry-tuned checkpoint used for the 0.41 model:
If a geometry-tuned DINOv3 ViT-L checkpoint exists, repeat only:
E0 winner-compatible baseline
and
best overall candidate
using ViT-L.
Do not rerun the entire grid.
Compare S vs L under identical sampling/head conditions.
Then test:
frozen
vs
last 2 blocks trainable
vs
last 4 blocks trainable or LoRA
only for the final architecture.
Use a backbone LR much smaller than the head LR.
Do not destroy geometry representations with aggressive full fine-tuning on ~251 videos.
24. Runtime constraint
Benchmark a private-style Stage-2 inference run on an L40S.
Record:
    • encoded frames/video;
    • backbone time;
    • temporal-head time;
    • refinement time;
    • peak VRAM;
    • projected total competition runtime.
Do not choose an architecture that violates the competition execution budget.
25. Reproducibility
Every run must save:
config
git commit
seed
fold
checkpoint
metrics.json
predictions.csv/json
training history
parameter count
runtime
peak VRAM
Use W&B if already configured, but also save local structured files so results do not depend on W&B availability.
26. Required experiment table
Generate a final table similar to:
ID
Backbone
Sampling
Temporal head
Event target
Category head
ENTRY
COLLISION
SIDE F1
EVASION F1
Score
Params
Runtime
Include both fixed-split and CV results.
Do not hide failed experiments.
27. Analysis requirements
After experiments, answer these questions explicitly:
    1. Is the remaining bottleneck primarily temporal modeling, event loss, spatial resolution, or backbone capacity?
    2. Does classification+displacement improve exact event localization?
    3. Does TemporalMaxer-lite outperform more complicated temporal models?
    4. Does strict FPS-blind training improve robustness?
    5. Which target contributes most to gains?
    6. Does event-conditioned pooling improve entry_side and/or evasion_space?
    7. Does coarse-to-fine refinement improve temporal accuracy enough to justify its inference cost?
    8. Is 7×10 spatial pooling throwing away useful geometry?
    9. Does ViT-L provide meaningful gain over ViT-S after controlling for the head?
    10. Which changes are statistically stable over folds/seeds?
28. Final model selection rule
Do not automatically choose the highest single validation score.
Prefer a candidate that has:
    • high mean CV score;
    • low variance;
    • improvement across multiple folds/seeds;
    • robustness under temporal resampling;
    • private-compatible FPS-blind inference;
    • acceptable L40S runtime.
Leaderboard score may be reported if I manually provide it, but DO NOT repeatedly optimize architecture based on leaderboard submissions.
29. Deliverables
At completion provide:
    • working experiment code;
    • configs for every experiment;
    • all result JSON/CSV files;
    • cached feature documentation;
    • best checkpoints;
    • private-style inference implementation;
    • runtime benchmark;
    • ablation tables;
    • plots;
    • failure cases;
    • exact reproduction commands;
    • final Markdown report.
Save the main report to something like:
reports/stage2_geometry_dino_spotting_experiments.md
30. Execution requirement
Do not stop after writing code.
Actually:
inspect → reproduce baseline → implement strict FPS-blind pipeline → run smoke tests → run E0/E1/E2/E3 screen → select candidates → run robust CV → test event-conditioned heads → test temporal ensemble/refinement → test spatial/layer variants if justified → test ViT-L if checkpoint exists → benchmark runtime → write report.
If compute becomes limiting, prioritize in this order:
E0 strict FPS-blind
→ E1 T-DEED-lite
→ E2 TemporalMaxer-lite
→ E4 event-conditioned categorical heads
→ E5 temporal multi-view
→ E6 coarse-to-fine
→ E3 ASFormer-lite
→ E7 spatial refinement
Do not fabricate unrun experiments.
At the end print:
    • best configuration;
    • fixed-split score;
    • CV mean ± std;
    • all four target metrics;
    • improvement over E0;
    • checkpoint path;
    • inference command;
    • projected L40S runtime;
    • report path;
    • recommendation for the next experiment.
