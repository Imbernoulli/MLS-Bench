#!/bin/bash
# Evaluate on setting: d=500, k=10, alpha=1e-3 (rich regime)
set -euo pipefail

# Resolve the FIXED wrapper next to this script BEFORE any cd (robust to a
# relative invocation path).
FIXED_ENTRY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/fixed_entry.py"

cd /workspace

# The harness re-stages this run's pre-generated input blobs before every
# evaluation, so they are treated as ephemeral (read+unlinked by the fixed
# wrapper before any editable code runs).
export MLSBENCH_EPHEMERAL_INPUTS=1

OUT_DIR="${OUTPUT_DIR:-${SAVE_PATH:-/tmp/mlsbench_opt_diagonal_net}}"
EXTRA_ARGS=()

if [[ "${MLS_BENCH_SMOKE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--smoke)
fi

# FIXED wrapper: preloads and unlinks this run's staged input blobs BEFORE
# importing the editable module, so top-level statements in the editable
# range never see them on disk.
python "$FIXED_ENTRY" \
  --module RAIN/opt_diagonal_net/custom_optimizer.py \
  --inputs-glob "RAIN/opt_diagonal_net/_inputs/d500_k10_a0p001_*.npz.b64" \
  --inject-module fixed_benchmark \
  --entry main \
  -- \
  --seed "${SEED:-42}" \
  --label "${ENV:-d500_k10_a1e3}" \
  --output-dir "$OUT_DIR" \
  --dim 500 \
  --sparsity 10 \
  --delta 0.5 \
  --alpha-init 0.001 \
  "${EXTRA_ARGS[@]}"
