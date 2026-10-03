# Best-method selection and checkpoint provenance

Audit date: 2026-10-03. The participant confirmed using the best leaderboard results already recorded in the repository. The stated finish is 22/321. No new leaderboard scores were supplied for the last experimental archives.

## Selection

| Stage | Retain | Evidence |
|---|---|---|
| 1 | `global_g1_threshold_0_50`, threshold .5 | Handoff's v3 result records 0.953197975; same checkpoint in v8/v10 |
| 2 | v8 `E4_sa_full + E2_sa_full + XN4_sa_full`, four seeds each | Handoff update gives .59293; Sep-29 leaderboard report still ranks v8 first |
| 3 | `V3_tcnssm_100ep`, EMA epoch 91, decoder ±.4/7° | Stage 3 consolidated report and handoff record .7475 in v10 |

The exact submitted filename/hash for the reported v8 score was not supplied in the handoff. R2's v8 archive has the corresponding runtime, backbone and twelve named refit weights; the saved per-member configs verify the family recipe. That is a verified recipe/package match, not proof of the exact uploaded file identity.

The combined best-stage recipe in this repository is a new archival composition. The 22/321 rank is not attributed to this reconstructed ZIP or to an inferred weighted score.

## Why later candidates were excluded

| Stage 2 recipe | Recorded leaderboard |
|---|---:|
| v5 | .5314 |
| v6 | .5277 |
| v7 | .5464 |
| **v8** | **.59293** |
| v8 + KDe | .5886 |
| v10 | .5590 |
| v12 | .5635 |
| v13 | .5609 |

The final E4t/FG2/FGk/ViT-B, actor-hazard, flow-grid and distilled packages have no higher measured leaderboard result in the inspected records. Local forecasts do not justify replacing v8. BATON+DriveDNA Stage 3 fell to .6572 and TCP/Openpilot variants lacked a superior recorded leaderboard result.

An important correction from direct config inspection: v8's retained full-data refits **do not use phase loss** (`phase_rep=none`, all phase weights zero). Earlier reports studied phase supervision, but their findings do not describe the saved winning refit settings.

## Checkpoints

The audit downloaded these two existing, publicly accessible R2 objects:

* `submissions/2026-09-27/submit_v8_robust-E4E2XN4sa_V3_acc0p5_steer5.zip`
* `submissions/2026-09-27/submit_v10_XCE4-XSE2-XSXN4_V3_acc0p4_steer7.zip`

Their Stage 1 checkpoint, Stage 3 checkpoint and geometry-adapted DINOv3 backbone were byte-identical. The v10 decoder override is retained; all required weights can therefore be restored from v8 alone. No new access settings were enabled and no R2 objects were deleted or moved.

| Asset | SHA-256 |
|---|---|
| Stage 1 `best.pt` | `5c1b5214972e85b6afa2afe1fb59103d3355f76ff42942ffdf886584c06bb669` |
| DINOv3 geometry backbone | `cc6b4cdc41f6bdb2c2ff6529d53ea55fdc37f078edf6f27378edfde123ea5b61` |
| Stage 3 V3 `best.pt` | `ffb3b8d4c53c7350c95ea6e61bad15adc3b7c3fa16268ccf7455b757cfd442ec` |

[The manifest](../checkpoints/manifest.json) pins the archive and all sixteen weight files: Stage 1, Stage 2 backbone, twelve Stage 2 members, Stage 3 V3, and SEA-RAFT-S. Large binaries remain in R2; the repository records their exact identity and provides a verified restore command.

## Original evidence

The following source records remain accessible at their original commit:

* [Handoff and leaderboard history](https://github.com/taehyeon-k/car-accident-competition/blob/2f065e3dd6e44d4f8d85c785c3d56a30c28f29c9/HANDOFF.md)
* [Sep-29 Stage 2 scored recipes](https://github.com/taehyeon-k/car-accident-competition/blob/2f065e3dd6e44d4f8d85c785c3d56a30c28f29c9/reports/stage2_lb_forecast_methodology.md)
* [Stage 3 V1–V3 consolidated report](https://github.com/taehyeon-k/car-accident-competition/blob/012883a417710f60db9963373148e34b592a75de/stage3/STAGE3_V1_TO_V3_REPORT.md)
* [v8 inference config](https://github.com/taehyeon-k/car-accident-competition/blob/2f065e3dd6e44d4f8d85c785c3d56a30c28f29c9/submission_tools/v8_stage2/config.json)

The official [competition overview](https://dacon.io/competitions/official/236753/overview/description) and [evaluation rules](https://dacon.io/competitions/official/236753/overview/evaluation) define the task; the project records, rather than public task documentation, establish the participant's scores.
