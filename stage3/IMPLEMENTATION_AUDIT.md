# Stage 3 v1.2 implementation audit

| Requirement | Implementation |
|---|---|
| External PTS to exact 10 Hz | `data/timing.py::decode_external_training_video` |
| DACON every decoded frame, fixed 0.1 s | `data/timing.py::decode_dacon_stage3_video`, `inference/predictor.py` |
| Canonical adapter contract | `data/adapters/base.py::Signals`, `ClipRecord` |
| BATON direct acceleration and steering | `data/adapters/baton.py::BatonAdapter` |
| Direct acceleration primaries | `data/targets.py::make_targets` |
| Ground-truth speed derivative auxiliaries | `data/targets.py::_speed_acceleration` |
| Prior focal default; optional GeoCalib | `geometry/calibration.py::estimate_focal` |
| Frozen SEA-RAFT-S and uncertainty | `flow/sea_raft.py::SeaRaftS` |
| Offline local asset enforcement | `flow/sea_raft.py::SeaRaftS.__init__` |
| 3-DoF rotation and derotation | `geometry/rotation.py`, `geometry/pipeline.py` |
| FOE | `geometry/foe.py::estimate_foe` |
| Dense tracks | `geometry/tracks.py::advect_scalar` |
| rho k=2/k=4 | `geometry/rho.py`, `geometry/pipeline.py` |
| Canonical ten channels | `geometry/canonical_grid.py::canonical_tensor` |
| Ordered 20-D vector | `geometry/physics_features.py::physics_vector` |
| Versioned target-independent cache | `data/cache.py::cache_key`, `cache_record` |
| Training-only robust normalization | `scripts/compute_statistics.py` |
| Motion CNN and attention pool | `model/motion_cnn.py` |
| Physics MLP and linear fusion | `model/physics_mlp.py`, `model/model.py` |
| One bidirectional dilated TCN | `model/tcn.py::TemporalConvNet` |
| Correct continuous heads | `model/heads.py::MotionHeads` |
| Correct weighted masked losses | `trainer/losses.py::stage3_loss` |
| Steering-angle category decoder | `trainer/decoder.py::steering_scores` |
| Potts Viterbi | `trainer/decoder.py::potts_viterbi` |
| Accelerate, bf16, EMA, cosine, resume | `run.py`, `trainer/trainer.py` |
| Validation metrics | `trainer/metrics.py`, `Trainer.validate` |
| Callable exact-schema inference | `inference/dacon.py::predict_stage3` |
| Root inference integration | `submission/inference.py::predict_stage3` |
| Submission packaging | `stage2/scripts/build_submission.py` |
| Runtime breakdown | `scripts/benchmark_inference.py` |
| Required synthetic and integration tests | `tests/` |

Known follow-up work is empirical: cache the full BATON sample, train the baseline,
calibrate acceleration/steering thresholds on route-disjoint validation, and run
the A0/F2/A1b/A2 experiments. ADAS-TO column and sign mappings remain unresolved
because that dataset release is absent. GeoCalib remains an optional lazy ablation;
the default and submitted pipeline uses the focal prior.
