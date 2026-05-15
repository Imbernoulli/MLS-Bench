#!/bin/bash
# Evaluate on large-scale setting: d=10000, k=50, delta=0.5, alpha=1.0
set -euo pipefail

cd /workspace

OUT_DIR="${OUTPUT_DIR:-${SAVE_PATH:-/tmp/mlsbench_opt_diagonal_net}}"
EXTRA_ARGS=()

if [[ "${MLS_BENCH_SMOKE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--smoke)
fi

python RAIN/opt_diagonal_net/custom_optimizer.py \
  --seed "${SEED:-42}" \
  --label "${ENV:-d10000_k50}" \
  --output-dir "$OUT_DIR" \
  --dim 10000 \
  --sparsity 50 \
  --sigma 0.0 \
  --delta 0.5 \
  --alpha-init 1.0 \
  --n-test 10000 \
  --grid-max 2000 \
  "${EXTRA_ARGS[@]}"
