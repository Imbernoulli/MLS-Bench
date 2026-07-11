#!/usr/bin/env bash
set -euo pipefail
export QA_DATA="${MLSBENCH_VERIFIER_DATA_ROOT:?}/extractive-qa/data"
python -u harness_question_inclusion.py --solution solution/question_inclusion.py --dataset mrqa_newsqa_validation.jsonl --seed "${SEED:-42}"
