#!/bin/bash
set -euo pipefail
MARKER=/workspace/data/stage3/baton_drivedna_final/cache_work/complete.json
while [ ! -s "${MARKER}" ]; do
  STATUS=$(supervisorctl status stage3_drivedna_cache || true)
  case "${STATUS}" in
    *EXITED*)
      # The initial cache runner predates the completion marker. Its success
      # line is emitted only after every worker exits successfully and all
      # manifest cache paths exist. The training wrapper checks paths again.
      if tail -n 1 /workspace/data/stage3/baton_drivedna_final/cache_supervisor.log | rg -q '^complete=5395$'; then
        echo "DriveDNA cache completed successfully (legacy runner)"
        break
      fi
      echo "DriveDNA cache stopped before completion: ${STATUS}" >&2
      exit 1
      ;;
    *FATAL*|*BACKOFF*)
      echo "DriveDNA cache stopped before completion: ${STATUS}" >&2
      exit 1
      ;;
  esac
  echo "waiting for DriveDNA caches: ${STATUS}"
  sleep 30
done
# User requested immediate training; use synchronous data loading to reduce RAM.
exec /workspace/car-accident/stage3/experiments/run_baton_drivedna_train.sh
