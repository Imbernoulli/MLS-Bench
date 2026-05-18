#!/bin/bash
set -euo pipefail

cd /workspace

OUT_DIR="${OUTPUT_DIR:-${SAVE_PATH:-/tmp/mlsbench_optimization_convex_concave}}"
EXTRA_ARGS=()

if [[ "${MLS_BENCH_SMOKE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--smoke)
fi

python RAIN/optimization_convex_concave/custom_strategy.py \
  --seed "${SEED:-42}" \
  --label "${ENV:-low-noise}" \
  --output-dir "$OUT_DIR" \
  --sigma-scale 0.1 \
  "${EXTRA_ARGS[@]}"
