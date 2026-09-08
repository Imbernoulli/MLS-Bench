#!/bin/bash
set -euo pipefail

cd /workspace
OUT_DIR="${OUTPUT_DIR:-${SAVE_PATH:-/tmp/mlsbench_optimization_bilevel}}"
mkdir -p "$OUT_DIR"

python penalized-bilevel-gradient-descent/mlsbench/custom_strategy.py \
  --mode toy \
  --seed "${SEED:-42}" \
  --label "${ENV:-toy-convergence}" \
  --output-dir "$OUT_DIR"
