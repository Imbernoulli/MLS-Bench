#!/bin/bash
set -euo pipefail

cd /workspace
OUT_DIR="${OUTPUT_DIR:-${SAVE_PATH:-/tmp/mlsbench_optimization_bilevel}}"
mkdir -p "$OUT_DIR"

python penalized-bilevel-gradient-descent/mlsbench/custom_strategy.py \
  --mode hyperclean \
  --net linear \
  --seed "${SEED:-42}" \
  --label "${ENV:-hyperclean-linear}" \
  --output-dir "$OUT_DIR"
