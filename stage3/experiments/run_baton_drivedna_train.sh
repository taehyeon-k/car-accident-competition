#!/bin/bash
set -euo pipefail
cd /workspace/car-accident
source /venv/main/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

MANIFESTS=/workspace/data/stage3/baton_drivedna_final/manifests
BASE=runs/stage3_v2/V3_tcnssm_100ep
ROOT=runs/stage3_data_scaling
NAME=Baton_DriveDNA_V3
OUT="${ROOT}/${NAME}"
if [ -s stage3/experiments/BATON_DRIVEDNA_REPORT.md ]; then
  echo "experiment report already exists; completed run left intact"
  exit 0
fi

python - <<'PY'
from pathlib import Path
from stage3.utils.config import read_jsonl
for name in ('train', 'val'):
    manifest = Path('/workspace/data/stage3/baton_drivedna_final/manifests') / f'{name}.jsonl'
    missing = [row['cache_path'] for row in read_jsonl(manifest) if not Path(row['cache_path']).is_file()]
    if missing:
        raise SystemExit(f'{len(missing)} {name} caches missing; first: {missing[0]}')
PY

mkdir -p "${ROOT}"
if [ ! -f "${OUT}/metrics.json" ]; then
  if [ -f "${OUT}/last.pt" ]; then
    python -m stage3.experiments.run --root "${ROOT}" --name "${NAME}" \
      --gpu-memory-fraction 0.55 --resume
  else
    python -m stage3.experiments.run --root "${ROOT}" --name "${NAME}" \
      --base "${BASE}/config.yaml" --gpu-memory-fraction 0.55 \
      --set data.manifest="${MANIFESTS}/train.jsonl" \
      --set data.val_manifest="${MANIFESTS}/baton_val.jsonl" \
      --set data.num_workers=2 --set data.prefetch_factor=1
  fi
fi

python -m stage3.experiments.evaluate_sources \
  --config /workspace/data/stage3/baton_baseline_eval/config.yaml \
  --checkpoint "${BASE}/best.pt" \
  --output "${ROOT}/baseline_sources.json" \
  --manifest baton_val="${MANIFESTS}/baton_val.jsonl" \
  --manifest drivedna_val="${MANIFESTS}/drivedna_val.jsonl" \
  --manifest combined_val="${MANIFESTS}/val.jsonl" \
  --manifest baton_train="${MANIFESTS}/baton_train.jsonl"

python -m stage3.experiments.evaluate_sources \
  --config "${OUT}/config.yaml" \
  --checkpoint "${OUT}/best.pt" \
  --output "${ROOT}/merged_sources.json" \
  --manifest baton_val="${MANIFESTS}/baton_val.jsonl" \
  --manifest drivedna_val="${MANIFESTS}/drivedna_val.jsonl" \
  --manifest combined_val="${MANIFESTS}/val.jsonl" \
  --manifest combined_train="${MANIFESTS}/train.jsonl"

python -m stage3.experiments.summarize_drivedna --root "${ROOT}" \
  --output stage3/experiments/BATON_DRIVEDNA_REPORT.md
