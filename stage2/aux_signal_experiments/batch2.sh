#!/bin/bash
# Exp 3: motion inputs, P2 loss (= M_motion recipe). M0 = current global motion (matched control), M1 = global + residual, M2 = residual only.
until grep -q BATCH1_DONE stage2/aux_signal_experiments/logs/batches.log 2>/dev/null; do sleep 15; done
export P=10; R=stage2/aux_signal_experiments/run.sh; F="0 1"; CV="0 1 2"
$R M0_global   "$F" "$CV" --base-loss p2 --motion global
$R M1_both     "$F" "$CV" --base-loss p2 --motion both
$R M2_residual "$F" "$CV" --base-loss p2 --motion residual
echo BATCH2_DONE >> stage2/aux_signal_experiments/logs/batches.log
