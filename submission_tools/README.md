# DACON submission

The standalone deliverable is `../submission/`. Its top level contains only
`inference.py`, `requirements.txt`, and `model/` with `stage1/`, `stage2/`, `stage3/`.
All entry points accept either the common data/model root or the stage-specific root.
Inference has no dependency on the training checkout, caches, labels, or internet.

- **Stage 1:** original `global_g1_threshold_0_50.zip` implementation and checkpoint,
  threshold 0.50. No changes to its preprocessing or mixed precision.
- **Stage 2:** learned FPS-independent tuned no-anchor DINOv3-S plus a small
  temporal head. It samples 128 evenly spaced relative positions from the entire
  sequence. No FPS or duration metadata is read. Short clips repeat positions;
  duplicate positions are encoded only once. Batched BF16 backbone, FP32 temporal
  head, original frame-number outputs, entry ≤ collision. The final head is trained
  on all 251 labeled videos. This replaces the random fallback.
- **Stage 3:** current run's EMA weights, checkpoint normalization and decoder.
  SEA-RAFT-S is bundled; flow is batched and geometry uses CUDA. Motion CNN runs
  in 32-frame chunks while the temporal model retains full clip context. Frame
  resizing occurs during decode with the identical OpenCV INTER_AREA operation,
  saving RAM. No frame skipping: all decoded frames receive both labels, including
  STOPPED frames. Uses fixed 0.1-second intervals, independent of video PTS.

## Final Stage 3 checkpoint

Stage 3 `best.pt` is deliberately absent until your run finishes. In the container:

```bash
cd /workspace/car-accident
cp runs/stage3/baseline_v1_2/best.pt submission/model/stage3/best.pt
python submission_tools/build_zip.py --output /workspace/outputs/submit.zip
```

The builder rejects missing checkpoints and packages files directly at ZIP root.
SEA-RAFT's weights/source are already included. Do not move the whole training run,
optimizer directory, data, feature caches or physics-statistics file: normalization
is stored in the Stage 3 checkpoint.

## Stage 2 training and limits

`train_fps_stage2.py` uses cached tuned-backbone features and sampling jitter.
Epoch selection used the original 201 training / 50 validation split over 40
candidate epochs. Epoch 5 had the best native-time competition-style score,
0.609989 (entry accuracy 0.40, collision 0.70, side Macro-F1 0.81993, evasion
Macro-F1 0.68). The final head was refit from initialization for 5 epochs on all
251 labels with the same learning-rate trajectory. The reported validation score
belongs to the selection model, **not an independent test of the final refit**.
Training took 101.6 seconds and peaked at 411.5 MiB allocated GPU memory. The
existing Stage 3 process was left running.

Sampling uses relative frame-number positions, not a seconds-based stride. Thus
FPS metadata and affine renumbering cannot change the visual sequence presented
to the model. Actual dropped frames can still remove evidence; accuracy is not
mathematically invariant to arbitrary resampling or clip truncation. The held-out
cached-feature stress test scores were 0.610 / 0.596 / 0.582 / 0.530 at sampling
strides 1 / 2 / 3 / 4. Very long clips have coarser event localization because only
128 positions are encoded. Known source FPS is used only for training loss widths
and evaluation, never as an input to the prediction network or submission sampler.

Historical models remain outside the ZIP in `learned_stage2/` (old FPS-dependent
model), `random_stage2/` (random fallback) and `fps_stage2/` (current training
artifacts, including the separate held-out selection checkpoint).

## Verification

Run in `car-accident-dev`, outside the repository working directory:

```bash
cd /tmp
python /workspace/car-accident/submission_tools/smoke_submission.py
python /workspace/car-accident/submission_tools/test_fps_stage2.py
```

The smoke test copies the package into a temporary directory, injects a temporary
snapshot of the current Stage 3 checkpoint, blocks outbound socket connections,
and exercises all three entry points on real-video/frame excerpts. It checks
schemas, labels, frame indices, and exact streamed-resize equivalence. It does not
install the temporary Stage 3 checkpoint into the deliverable. Results are written
alongside these tools. Excerpt timings do not establish the hidden full-dataset
60-minute limit, since DACON does not disclose its test-set size.

Official contract: https://dacon.io/competitions/official/236753/overview/evaluation
Input layouts: https://dacon.io/competitions/official/236753/data
