#!/usr/bin/env bash
set -euo pipefail
export QA_DATA="${MLSBENCH_VERIFIER_DATA_ROOT:?}/extractive-qa/data"
python -u harness_span_decoding.py --solution solution/span_decoding.py --dataset mrqa_newsqa_validation.jsonl --seed "${SEED:-42}"
