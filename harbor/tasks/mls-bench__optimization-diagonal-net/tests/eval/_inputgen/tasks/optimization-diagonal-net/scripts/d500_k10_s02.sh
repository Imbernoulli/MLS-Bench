#!/bin/bash
# Evaluate on setting: d=500, k=10, sigma=0.2
set -euo pipefail

cd /workspace

# The harness re-stages this run's pre-generated input blobs before every
# evaluation, so the runner treats them as ephemeral (deleted right after
# loading, before any editable code runs).
export MLSBENCH_EPHEMERAL_INPUTS=1

OUT_DIR="${OUTPUT_DIR:-${SAVE_PATH:-/tmp/mlsbench_opt_diagonal_net}}"
EXTRA_ARGS=()

if [[ "${MLS_BENCH_SMOKE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--smoke)
fi

python RAIN/opt_diagonal_net/custom_optimizer.py \
  --seed "${SEED:-42}" \
  --label "${ENV:-d500_k10_s02}" \
  --output-dir "$OUT_DIR" \
  --dim 500 \
  --sparsity 10 \
  --sigma 0.2 \
  --delta 0.5 \
  "${EXTRA_ARGS[@]}"
