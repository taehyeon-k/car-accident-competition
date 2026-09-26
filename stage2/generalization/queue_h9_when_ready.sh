#!/bin/bash
# Add H9 (unlabelled consistency) jobs to the front of the queue once the unlabelled pool is fully extracted.
cd /workspace/car-accident
until grep -q "^done" stage2/generalization/unl_expand.log; do sleep 30; done
/venv/main/bin/python - <<'PY'
import sys; sys.path.insert(0, "/workspace/car-accident")
from stage2.generalization.jobs_util import loso_unl, cv, add
S=[0,1,2]; A="--motion both --stride-aug 0.5,0.25,0.25 --unl-consistency 0.3"
add(cv("E4_sa_unl03",S,A)+loso_unl("LOSO_E4_sa_unl03",S,A), front=True)
PY
echo "H9 queued $(date)" >> stage2/generalization/queue.out
