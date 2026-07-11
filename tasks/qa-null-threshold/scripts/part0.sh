#!/usr/bin/env bash
set -euo pipefail
export QA_DATA="${MLSBENCH_VERIFIER_DATA_ROOT:?}/extractive-qa/data"
python -u harness_null_threshold.py --solution solution/null_threshold.py --dataset squad2_validation_part0.jsonl --seed "${SEED:-42}"
