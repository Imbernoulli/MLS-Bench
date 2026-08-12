#!/bin/bash
# Evaluate on setting: d=500, k=10, sigma=0.2
set -euo pipefail

cd /workspace

# Inputs for this run are re-materialized by apply.py below, so the runner may
# treat them as ephemeral (load into memory, then delete before editable code
# runs).
export MLSBENCH_EPHEMERAL_INPUTS=1
# Splice the fixed runner blocks into stale baked workspace images (anchored
# exact-match replaces; no-op when the workspace is already up to date).
source "$(dirname "${BASH_SOURCE[0]}")/_runtime_patch.sh"

OUT_DIR="${OUTPUT_DIR:-${SAVE_PATH:-/tmp/mlsbench_opt_diagonal_net}}"
EXTRA_ARGS=()

if [[ "${MLS_BENCH_SMOKE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--smoke)
fi

python "/tests/eval/_inputgen/apply.py" "optimization-diagonal-net" /workspace
python RAIN/opt_diagonal_net/custom_optimizer.py \
  --seed "${SEED:-42}" \
  --label "${ENV:-d500_k10_s02}" \
  --output-dir "$OUT_DIR" \
  --dim 500 \
  --sparsity 10 \
  --sigma 0.2 \
  --delta 0.5 \
  "${EXTRA_ARGS[@]}"
