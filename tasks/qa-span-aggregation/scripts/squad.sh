#!/usr/bin/env bash
set -euo pipefail
export QA_DATA="${MLSBENCH_VERIFIER_DATA_ROOT:?}/extractive-qa/data"
python -u harness_span_aggregation.py --solution solution/span_aggregation.py --dataset mrqa_squad_validation.jsonl --seed "${SEED:-42}"
