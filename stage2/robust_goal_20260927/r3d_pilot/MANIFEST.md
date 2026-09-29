# Frozen-video feature Stage2 pilot

**Hypothesis:** a temporal representation pretrained on video may expose the early motion of a small intruding vehicle that the frozen spatial DINO features and 100-d global/residual motion miss. This is a different feature source from the existing object crops, lane geometry, and sparse residual motion.

**Control:** E4_sa fold0 seed0, native/half/third-rate stride mix 0.5/0.25/0.25, original frozen DINO tokens + global/residual motion, standard direct heads and decoder.

**Candidate:** same split, seed, training recipe and decoder, adding Kinetics-400 pretrained R3D-18 layer2/layer3 spatial-mean features through a separately normalized gated projection. R3D reads the exact adaptively selected RGB frames for each rate view; reduced-rate features are recomputed from that view rather than sampled from native-rate features. Its checkpoint is cached under `../pretrained/`. Source, FPS, timestamps, duration, and event labels never enter the feature extractor or inference input.

**Screen:** duplicate-clean original fold0 validation at native, half, and third rate, source-wise score, ENTRY on gaps >=1.5 s, and measured Stage2 feature-extraction latency. Advance only if the candidate gains >=0.015 total score at both native and third rate, has no source score loss >0.03, and improves long-gap ENTRY. If it fails, stop this feature family after the one-fold pilot. If it passes, run all five folds and at least three seeds, source holdouts, crop tests, and fixed-size ensemble comparison before packaging. No leaderboard claim from this pilot.

**Isolation:** all feature caches and model outputs live under `stage2/robust_goal_20260927/r3d_pilot/`. The extractor yields before each clip whenever another GPU process is present. It does not modify shared caches, training code, checkpoints, or running jobs. The R3D inference overhead is a deployment cost and must be justified by a meaningful gain.
