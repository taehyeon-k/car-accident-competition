#!/bin/bash
set -euo pipefail
cd /workspace/car-accident
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
OUT=runs/stage3_data_scaling/drivedna_interim
for MODEL in baseline combined; do
  if [ -s "${OUT}/${MODEL}.json" ]; then continue; fi
  if [ "${MODEL}" = baseline ]; then
    CFG=/workspace/data/stage3/baton_baseline_eval/config.yaml
    CKPT=runs/stage3_v2/V3_tcnssm_100ep/best.pt
  else
    CFG=runs/stage3_data_scaling/Baton_DriveDNA_V3/config.yaml
    CKPT="${OUT}/combined_snapshot.pt"
  fi
  echo "Evaluating ${MODEL} on full DriveDNA validation"
  /venv/main/bin/python -m stage3.experiments.evaluate_sources \
    --config "${CFG}" --checkpoint "${CKPT}" --output "${OUT}/${MODEL}.json" \
    --manifest drivedna_val=/workspace/data/stage3/baton_drivedna_final/manifests/drivedna_val.jsonl \
    --accel-threshold .25 --device cuda --gpu-memory-fraction .10 \
    --batch-size 1 --num-workers 0 --chunk-frames 16
done
