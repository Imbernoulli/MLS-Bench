#!/bin/bash
# Evaluate on setting: d=200, k=5, sigma=0.1
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

_staged_list="$(mktemp /tmp/mlsb_staged.XXXXXX)"
# EXIT backstop: delete exactly the blobs apply.py staged for THIS eval,
# even when the runner crashes/times out before its in-process scrub —
# a crashed eval must not leave its withheld inputs readable for the rest
# of the wave. Normally a no-op (the runner scrubs right after loading).
trap 'xargs -r rm -f -- < "$_staged_list" 2>/dev/null; rm -f "$_staged_list"' EXIT
python "/tests/eval/_inputgen/apply.py" "optimization-diagonal-net" /workspace --list-out "$_staged_list"
python RAIN/opt_diagonal_net/custom_optimizer.py \
  --seed "${SEED:-42}" \
  --label "${ENV:-d200_k5_s01}" \
  --output-dir "$OUT_DIR" \
  --dim 200 \
  --sparsity 5 \
  --sigma 0.1 \
  --delta 0.5 \
  "${EXTRA_ARGS[@]}"
