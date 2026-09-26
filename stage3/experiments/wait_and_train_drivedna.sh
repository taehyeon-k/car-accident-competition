#!/bin/bash
set -euo pipefail
MARKER=/workspace/data/stage3/baton_drivedna_final/cache_work/complete.json
while [ ! -s "${MARKER}" ]; do
  STATUS=$(supervisorctl status stage3_drivedna_cache || true)
  case "${STATUS}" in
    *FATAL*|*BACKOFF*|*EXITED*)
      echo "DriveDNA cache stopped before completion: ${STATUS}" >&2
      exit 1
      ;;
  esac
  echo "waiting for DriveDNA caches: ${STATUS}"
  sleep 30
done
exec /workspace/car-accident/stage3/experiments/run_baton_drivedna_train.sh
