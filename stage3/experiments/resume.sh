#!/bin/bash
# usage: resume.sh NAME   — continue an interrupted run from runs/stage3_v2/NAME/last.pt
cd /workspace/car-accident; source /venv/main/bin/activate >/dev/null
N=$1
python -m stage3.experiments.run --name $N --resume >> runs/stage3_v2/logs/$N.log 2>&1
echo "EXIT $? $N" >> runs/stage3_v2/logs/$N.log
