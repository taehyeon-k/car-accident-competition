#!/bin/bash
# usage: exp.sh NAME [--set k=v ...]
cd /workspace/car-accident; source /venv/main/bin/activate >/dev/null
N=$1; shift
python -m stage3.experiments.run --name $N \
  --set data.manifest=/workspace/data/stage3/manifests_raw/train.jsonl \
  --set data.val_manifest=/workspace/data/stage3/manifests_raw/val.jsonl \
  --set data.num_workers=${WORKERS:-4} "$@" > runs/stage3_v2/logs/$N.log 2>&1
echo "EXIT $? $N" >> runs/stage3_v2/logs/$N.log
