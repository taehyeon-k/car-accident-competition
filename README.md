# Car Accident Competition

**22nd out of 321 teams — top 6.9%.** My result in DACON's [블랙박스 영상 기반 지능형 고의사고 분석 모델 AI 경진대회](https://dacon.io/competitions/official/236753/overview/description) (2026).

This project analyzes dashcam footage across three tasks: detecting re-recorded video, locating accident events, and predicting acceleration and steering categories. This repository preserves the **best methods supported by recorded competition leaderboard results**, their inference code, and checksum-pinned checkpoints.

## Results

| Stage | Retained method | Best recorded leaderboard score |
|---|---|---:|
| 1 — re-recording detection | ConvNeXt-Tiny local/global RGB + native-rate burst profiles (`global_g1_threshold_0_50`) | 0.953197975 |
| 2 — accident events and attributes | v8: geometry-adapted DINOv3-S + temporal pyramid + global/residual motion; E4/E2/XN4 ensemble with temporal-rate augmentation | **0.59293** |
| 3 — vehicle behavior | V3: SEA-RAFT-S + motion/physics features + gated dual-TCN/Bi-SSM; acceleration ±0.4, steering 7° | **0.7475** |

Rank is participant-reported. Scores are the stage-specific results recorded in the project reports, not an overall score. These stage bests came from different submissions. The retained combination has no separately recorded leaderboard result and is not claimed to be the exact final ranked submission. [Selection evidence and checkpoint provenance](docs/selection.md).

## Competition

The competition is a three-stage, offline code-submission task hosted by the Ministry of the Interior and Safety and NIA, organized by the National Forensic Service, and operated by DACON.

| Stage | Input | Output |
|---|---|---|
| 1 | Video | `ORIGINAL` or `RERECORDED` |
| 2 | Accident frame sequence | ENTRY frame, COLLISION frame, entry side, evasion-space availability |
| 3 | Driving video | Acceleration and steering categories at 0.1-second intervals |

Inference runs without internet access on an L40S evaluation server. Model loading and inference do not download weights; checkpoints must be restored before packaging. See the [official task description](https://dacon.io/competitions/official/236753/overview/description) and [evaluation rules](https://dacon.io/competitions/official/236753/overview/evaluation).

## Methods

**Stage 1.** A ConvNeXt-Tiny encoder combines local image patches and global frames with temporal burst statistics. Attention pools the frame and burst representations; gated fusion produces the re-recording probability. The retained checkpoint uses a 0.5 decision threshold.

**Stage 2.** A frozen, geometry-adapted DINOv3 ViT-S/16 encodes sampled frames into a 7×10 token grid. Frame-count-adaptive sampling supports short and long inputs without requiring FPS or duration metadata. A temporal pyramid combines appearance with camera motion and residual motion after camera alignment. The v8 ensemble has 12 learned members: four seeds each of E4, E2, and XN4. E2 adds boundary supervision; XN4 uses weakly supervised NEXAR expansion. Direct event supervision and training at native, half and third temporal rates improve event localization under changes in frame spacing. The saved winning configs disable phase loss. Joint decoding enforces ENTRY ≤ COLLISION and snaps COLLISION to a native-frame motion peak.

**Stage 3.** Frozen SEA-RAFT-S estimates optical flow and confidence. Geometry derives camera rotation, expansion, focus-of-expansion and acceleration/speed-ratio features. A MotionCNN and physics MLP feed a gated dual-dilated TCN plus bidirectional state-space model. Physical regression heads predict acceleration, speed, stop probability and steering angle. Thresholds and Potts/Viterbi decoding convert these into competition categories. The winning V3 checkpoint uses EMA weights from epoch 91 of a 100-epoch BATON training run. [Architecture details](docs/methods.md).

## Run

Python 3.12 and PyTorch 2.8 / torchvision 0.23 match the retained submission environment. Use CUDA builds for GPU inference.

```bash
pip install -r requirements.txt
python scripts/fetch_checkpoints.py
python scripts/build_submission.py
```

Checkpoint restoration downloads one verified v8 archive from the existing R2 bucket and extracts only the 16 required weight files. Git tracks the manifest and source; the binary weights remain in R2. Every archive and checkpoint has a recorded SHA-256. An existing archive can be reused with `--archive /path/to/v8.zip`.

The built ZIP contains `inference.py`, `requirements.txt`, and `model/` at its root. The competition calls these functions:

```python
from inference import predict_stage1, predict_stage2, predict_stage3

stage1 = predict_stage1("/path/to/data", "model")
stage2 = predict_stage2("/path/to/data", "model")
stage3 = predict_stage3("/path/to/data", "model")
```

Stage 2 expects numbered frames in `stage2/images/<ID>/`; Stage 1 and Stage 3 accept videos through their retained adapters. A nested stage directory or a direct stage root can be supplied.

## Repository

| Path | Purpose |
|---|---|
| `inference.py` | Three competition entry points |
| `model/stage1/` | Retained re-recording detector |
| `model/stage2/` | v8 runtime, pyramid models, sampling and DINOv3 source |
| `model/stage3/` | V3 runtime, motion/physics/temporal models and SEA-RAFT source |
| `checkpoints/manifest.json` | Exact checkpoint names, sizes, hashes and R2 location |
| `scripts/` | Checkpoint restoration and offline ZIP creation |
| `training/` | Per-stage training entry points, recovered settings and saved v8 configs |
| `stage2/` | Original v8 trainer, preprocessing and its shared source dependencies |
| `docs/` | Method description, selection evidence, cleanup and verification records |

The active tree excludes datasets, feature caches, training outputs, campaign queues, duplicate submission packages and superseded experiment reports. The retained Stage 2 and Stage 3 training code shows how the winning methods train on preprocessed inputs. Stage 1 has a clearly labeled reconstructed trainer because its original trainer was absent from the submitted assets. See [training instructions and data contracts](docs/training.md). Broader experiment history remains recoverable through the commits listed in [the cleanup record](docs/cleanup.md).

## What improved performance

Temporal-rate augmentation moved Stage 2 from v7 **0.5464** to v8 **0.59293**, despite only a small native-validation change. Later v10/v12 variants improved local validation but scored **0.5590/0.5635** on the leaderboard. This made measured transfer performance the criterion for retention.

For Stage 3, the TCN+SSM temporal model and longer schedule were the main architectural gains. Decoder calibration further improved the same V3 weights. Training V3 with BATON+DriveDNA reached higher local validation but fell to **0.6572** on the leaderboard; it is excluded from the retained method.

## Third-party components

Bundled DINOv3 source retains its [license](model/stage2/vendor/LICENSE.md). Bundled SEA-RAFT source retains its [BSD-3-Clause license](model/stage3/pretrained/sea_raft/source/LICENSE). Their licenses also govern the corresponding pretrained assets. The project does not redistribute training datasets.
