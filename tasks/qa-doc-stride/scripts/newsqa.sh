#!/usr/bin/env bash
set -euo pipefail
export QA_DATA="${MLSBENCH_VERIFIER_DATA_ROOT:?}/extractive-qa/data"
python -u harness_doc_stride.py --solution solution/doc_stride.py --dataset mrqa_newsqa_validation.jsonl --seed "${SEED:-42}"
