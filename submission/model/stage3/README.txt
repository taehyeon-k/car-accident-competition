Stage 3: SEA-RAFT-S flow -> geometry/physics features -> MotionCNN + physics MLP -> TCN+SSM hybrid temporal model
-> continuous heads -> Potts/Viterbi decoder, one label pair per decoded frame (fixed 0.1 s intervals).
best.pt: runs/stage3_v2/V3_tcnssm_100ep/best.pt (finished 100-epoch run, best epoch 91, validation competition score 0.7954;
  EMA weights, config, physics normalization; optimizer/scheduler state removed).
decoder_overrides.json: steering threshold_deg 1.5 (was 5.0), fitted on DACON OPEN labels; emission scale unchanged.
code/stage3: current repository stage3 package. SEA-RAFT source/weights under pretrained/sea_raft/.
