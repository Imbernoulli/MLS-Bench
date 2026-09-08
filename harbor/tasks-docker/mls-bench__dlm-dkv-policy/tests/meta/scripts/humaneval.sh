#!/bin/bash
set -euo pipefail
if [ -d /workspace ]; then
  cd /workspace
fi
DLLM_ROOT="${DLLM_CACHE_DIR:-dLLM-cache}"
python3 "${DLLM_ROOT}/custom_dlm_eval.py" --workload humaneval --regime final --seed "${SEED:-42}"
