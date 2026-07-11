#!/usr/bin/env bash
set -euo pipefail
export QA_DATA="${MLSBENCH_VERIFIER_DATA_ROOT:?}/extractive-qa/data"
python -u harness_n_best.py --solution solution/n_best.py --dataset mrqa_newsqa_validation.jsonl --seed "${SEED:-42}"
