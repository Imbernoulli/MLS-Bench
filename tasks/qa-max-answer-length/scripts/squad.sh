#!/usr/bin/env bash
set -euo pipefail
export QA_DATA="${MLSBENCH_VERIFIER_DATA_ROOT:?}/extractive-qa/data"
python -u harness_max_answer_length.py --solution solution/max_answer_length.py --dataset mrqa_squad_validation.jsonl --seed "${SEED:-42}"
