#!/bin/bash
set -uo pipefail
: "${MLSBENCH_VERIFIER_DATA_ROOT:?MLSBENCH_VERIFIER_DATA_ROOT is required}"
export NLI_EVAL_DATA="${MLSBENCH_VERIFIER_DATA_ROOT}/natural-language-inference"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd /workspace/natural-language-inference
python harness_class_weighting.py --solution solution/class_weighting.py --seed "${SEED:-42}"
rc=$?
printf 'NLI_COMMAND_DONE rc=%s\n' "$rc"
exit "$rc"
