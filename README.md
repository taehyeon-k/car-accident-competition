# Car Accident Competition

**22nd out of 321 teams — top 6.9%.** My result in DACON's [블랙박스 영상 기반 지능형 고의사고 분석 모델 AI 경진대회](https://dacon.io/competitions/official/236753/overview/description) (2026).

This project analyzes dashcam footage across three tasks: detecting re-recorded video, locating accident events, and predicting acceleration and steering categories. This repository preserves the **best methods supported by recorded competition leaderboard results**, their inference code, and checksum-pinned checkpoints.

## Results

| Stage | Retained method | Best recorded leaderboard score |
|---|---|---:|
| 1. re-recording detection | ConvNeXt-Tiny local/global RGB + native-rate burst profiles | **0.953197975** |
| 2. accident events and attributes | Geometry-adapted DINOv3-S + temporal pyramid + global/residual motion; E4/E2/XN4 ensemble with temporal-rate augmentation | **0.59293** |
| 3. vehicle behavior | SEA-RAFT-S + motion/physics features + gated dual-TCN/Bi-SSM; acceleration ±0.4, steering 7° | **0.7475** |

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

### Stage 1 — Re-recording detection

- **Visual features:** ConvNeXt-Tiny examines local image patches and whole frames to capture both fine detail and the overall image.
- **Temporal features:** Burst statistics summarize changes across short sequences of frames.
- **Fusion:** Attention pooling summarizes the frame and burst features; gated fusion combines them into a re-recording probability.
- **Decision:** A probability threshold of **0.5** separates `ORIGINAL` from `RERECORDED`.

### Stage 2 — Accident events and attributes

- **Frame sampling:** Sampling adapts to the frame count, supporting short and long inputs without FPS or duration metadata.
- **Appearance:** A frozen, geometry-adapted **DINOv3 ViT-S/16** represents each sampled frame as a **7×10 grid of visual features**.
- **Motion:** The model uses camera motion and residual motion—the movement remaining after camera alignment—alongside appearance features.
- **Temporal model:** A temporal pyramid combines information across different time scales to locate ENTRY and COLLISION.
- **Ensemble:** The final model averages **12 models**, with four training seeds each for **E4, E2, and XN4**. E2 adds boundary supervision; XN4 adds weakly supervised NEXAR examples.
- **Training:** Direct event supervision and native, half, and third temporal-rate augmentation help the model handle changes in frame spacing. **Phase loss is disabled** in the saved winning configurations.
- **Decoding:** Joint decoding enforces **ENTRY ≤ COLLISION** and refines COLLISION by snapping it to a motion peak in the original frame sequence.

### Stage 3 — Vehicle behavior

- **Motion extraction:** Frozen **SEA-RAFT-S** estimates optical flow (image movement between frames) and confidence.
- **Physics features:** Geometry derives camera rotation, expansion, focus of expansion, and acceleration/speed-ratio features.
- **Feature fusion:** A MotionCNN processes motion maps, while a physics MLP processes the geometry-derived features.
- **Temporal model:** A gated dual-dilated **TCN** and a bidirectional **state-space model** track how these features change over time.
- **Prediction:** Regression heads estimate acceleration, speed, stop probability, and steering angle.
- **Decoding:** Thresholds convert predictions into competition categories; **Potts/Viterbi sequence decoding** smooths the labels over time. The retained settings use acceleration **±0.4** and steering **7°**.
- **Checkpoint:** The winning model uses **EMA weights from epoch 91** of a **100-epoch BATON training run**.

See [architecture details](docs/methods.md) for the full technical description.

## Dataset

### Stage 1 — Re-recording detection

- **Dataset:** My own captured dataset, containing approximately **2 hours of video clips**.
- **Devices:** **LG Gram** and **MacBook Air** laptops, and **Samsung Galaxy S20, A13, and S24** phones.
- **Task:** Original-versus-re-recorded video classification. The reconstructed trainer accepts clips labeled `ORIGINAL` or `RERECORDED`.

The dataset description is author-provided. The original trainer was unavailable; the retained Stage 1 trainer is a reconstruction using the submitted model, preprocessing, and recovered settings.

### Stage 2 — Accident events and attributes

The labeled accident dataset contains **349 videos** with ENTRY, COLLISION, entry-side, and evasion-space annotations:

| Source | Labeled videos |
|---|---:|
| AI-Hub car-to-car footage | 88 |
| CCD (Car Crash Dataset) | 83 |
| NEXAR | 80 |
| MM-AU | 97 |
| CausalCrash | 1 |
| **Total** | **349** |

- **Evaluation:** Experiments used a 279/70 fixed train/validation split and five-fold cross-validation. The best Stage 2 models were then refitted on all **349 labeled videos**.
- **Additional supervision:** The **XN4 family** also used **670 additional NEXAR clips** with weak labels: collision timing from public event timestamps and teacher-generated ENTRY labels. These clips do not have the complete manual annotations of the 349-video set.
- **Backbone adaptation:** Before training the event models, the geometry-adapted DINOv3 backbone used **BDD100K, TuSimple, local CCD/NEXAR/AI-Hub footage, and BATON dashcam footage**. This step learned road geometry and motion; it did not use the four Stage 2 accident labels.

Counts and training roles are recorded in the archived [Stage 2 dataset review](https://github.com/taehyeon-k/car-accident-competition/blob/2f065e3dd6e44d4f8d85c785c3d56a30c28f29c9/stage2/STAGE2_EXPERIMENTS_DATASETS_LEADERBOARD_REVIEW_2026-09-27.md) and [backbone adaptation report](https://github.com/taehyeon-k/car-accident-competition/blob/2f065e3dd6e44d4f8d85c785c3d56a30c28f29c9/reports/stage2_geometry_dino_pretraining.md).

### Stage 3 — Vehicle behavior

- **Dataset:** **BATON-Sample**, using dashcam videos (`qcamera.mp4`) paired with vehicle CAN signals (`vehicle_dynamics.csv`).
- **Size:** **41 route folders**, divided into **2,159 thirty-second segments**—about **17.9 hours** in total.
- **Split:** **1,693 training segments from 33 routes** and **466 validation segments from 8 routes**. Related route recordings can cover the same roads across the split.
- **Targets:** Synchronized speed, longitudinal acceleration, and steering-wheel angle, aligned to a **10 Hz** video timeline.
- **Retained model:** The best Stage 3 checkpoint was trained on **BATON-Sample only**. The later BATON + DriveDNA experiment scored lower on the leaderboard.

See the archived [Stage 3 data and experiment report](https://github.com/taehyeon-k/car-accident-competition/blob/012883a417710f60db9963373148e34b592a75de/stage3/STAGE3_V1_TO_V3_REPORT.md) for the recorded split and preprocessing. Datasets and generated caches are not redistributed in this repository; see [training instructions](docs/training.md) for the required input formats.

## Run

Python 3.12 and PyTorch 2.8 / torchvision 0.23 match the retained submission environment. Use CUDA builds for GPU inference.

```bash
pip install -r requirements.txt
python scripts/fetch_checkpoints.py
python scripts/build_submission.py
```

The 16 required weight files are available in the [best-model checkpoint release](https://github.com/taehyeon-k/car-accident-competition/releases/tag/best-models-v1). Download the three stage checkpoint ZIPs and extract them into the repository root. Alternatively, `fetch_checkpoints.py` downloads the verified source archive from Cloudflare R2 and restores the same weights. Git tracks the source and checkpoint manifest; every checkpoint has a recorded SHA-256. An existing R2 archive can be reused with `--archive /path/to/checkpoints.zip`.

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
| `model/stage1/` | The best stage1 model's re-recording detector |
| `model/stage2/` | The best stage2 model's runtime, pyramid models, sampling and DINOv3 source |
| `model/stage3/` | The best stage3 model's runtime, motion/physics/temporal models and SEA-RAFT source |
| `checkpoints/manifest.json` | Exact checkpoint names, sizes, hashes|
| `scripts/` | Checkpoint restoration and offline ZIP creation |
| `training/` | Per-stage training entry points, recovered settings and saved configs |
| `stage2/` | Original trainer, preprocessing and its shared source dependencies |
| `docs/` | Method description, selection evidence, cleanup and verification records |

`training/stage2/` provides the command-line entry point and saved configurations. It imports the actual trainer, model helpers, data loading, and preprocessing from `stage2/`; deleting `stage2/` would break Stage 2 training. The inference code is separately packaged under `model/stage2/`.

The active tree excludes datasets, feature caches, training outputs, campaign queues, duplicate submission packages and superseded experiment reports. The retained Stage 2 and Stage 3 training code shows how the winning methods train on preprocessed inputs. Stage 1 has a clearly labeled reconstructed trainer because its original trainer was absent from the submitted assets. See [training instructions and data contracts](docs/training.md). Broader experiment history remains recoverable through the commits listed in [the cleanup record](docs/cleanup.md).

## What improved performance

Temporal-rate augmentation moved Stage 2 from **0.5464** to **0.59293**, despite only a small native-validation change. Later model variants improved local validation but scored **0.5590/0.5635** on the leaderboard. This made measured transfer performance the criterion for retention.

For Stage 3, the TCN+SSM temporal model and longer schedule were the main architectural gains. Decoder calibration further improved the same weights. Training the best method with BATON+DriveDNA reached higher local validation but fell to **0.6572** on the leaderboard; it is excluded from the retained method.

## Third-party components

Bundled DINOv3 source retains its [license](model/stage2/vendor/LICENSE.md). Bundled SEA-RAFT source retains its [BSD-3-Clause license](model/stage3/pretrained/sea_raft/source/LICENSE). Their licenses also govern the corresponding pretrained assets. The project does not redistribute training datasets.
