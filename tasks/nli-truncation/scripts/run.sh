#!/bin/bash
set -uo pipefail
: "${MLSBENCH_VERIFIER_DATA_ROOT:?MLSBENCH_VERIFIER_DATA_ROOT is required}"
export NLI_EVAL_DATA="${MLSBENCH_VERIFIER_DATA_ROOT}/natural-language-inference"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd /workspace/natural-language-inference
python harness_truncation.py --solution solution/truncation.py --seed "${SEED:-42}"
rc=$?
printf 'NLI_COMMAND_DONE rc=%s\n' "$rc"
exit "$rc"
