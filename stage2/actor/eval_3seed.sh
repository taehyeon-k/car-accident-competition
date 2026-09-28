#!/bin/bash
# 3-seed validation of D4Ro / D5H (waits for all 15 checkpoints of each), plus seeds 0-1 recipes for the LB forecast
cd /workspace/car-accident; L=stage2/actor/results/s2exp_3seed.log; P=/venv/main/bin/python; export OBJ_CACHE=stage2/objtrack/cache_objlane_full
until [ $(ls stage2/generalization/results/{D4Ro,D5H}/cv/fold*_seed[012]/checkpoint.pt 2>/dev/null | wc -l) -ge 30 ]; do sleep 30; done
for a in D4Ro D5H D4Ro+D5H E4_sa+E2_sa+XN4_sa+D4Ro+D5H; do $P -m stage2.generalization.entry_suite $a --seeds 0 1 2 --tag s012 2>&1 | grep -v Warn | grep -v "^$" >> $L; done
for a in D4Ro+D5H E4_sa+E2_sa+XN4_sa+D4Ro+D5H E4_sa+E2_sa+XN4_sa+ODS E4_sa+E2_sa+XN4_sa+D4Ro E4_sa+E2_sa+XN4_sa+D5H; do $P -m stage2.generalization.entry_suite $a --seeds 0 1 --tag s01 2>&1 | grep -v Warn | grep -v "^$" >> $L; done
echo EVAL_DONE >> $L
