#!/usr/bin/env bash
set -euo pipefail
export QA_DATA="${MLSBENCH_VERIFIER_DATA_ROOT:?}/extractive-qa/data"
python -u harness_max_seq_len.py --solution solution/max_seq_len.py --dataset mrqa_naturalquestions_validation.jsonl --seed "${SEED:-42}"
