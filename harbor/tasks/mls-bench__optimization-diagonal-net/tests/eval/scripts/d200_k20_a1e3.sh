#!/bin/bash
# Evaluate on setting: d=200, k=20, alpha=1e-3 (rich regime)
set -euo pipefail

cd /workspace

# Inputs for this run are PIPED to the fixed wrapper as JSON (apply.py
# --emit-json -> fixed_entry.py --inputs-json-stdin) and NEVER touch the
# shared workspace filesystem: Harbor runs the setting x seed wave
# CONCURRENTLY, so an on-disk staging window would be readable by a sibling
# eval's agent code. MLSBENCH_EPHEMERAL_INPUTS=1 marks them ephemeral.
export MLSBENCH_EPHEMERAL_INPUTS=1
_EVAL_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OUT_DIR="${OUTPUT_DIR:-${SAVE_PATH:-/tmp/mlsbench_opt_diagonal_net}}"
EXTRA_ARGS=()

if [[ "${MLS_BENCH_SMOKE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--smoke)
fi

_staged_list="$(mktemp /tmp/mlsb_staged.XXXXXX)"
# EXIT backstop: no-op with --emit-json (nothing is staged to disk); covers
# a version-skewed apply.py that stages the old way — the wrapper's
# --inputs-glob pass unlinks any such stragglers before importing the module.
trap 'xargs -r rm -f -- < "$_staged_list" 2>/dev/null; rm -f "$_staged_list"' EXIT
# FIXED wrapper: preloads this run's staged input blobs (from stdin; the disk
# glob is only an unlink backstop) BEFORE importing the editable module.
python "/tests/eval/_inputgen/apply.py" "optimization-diagonal-net" /workspace \
    --emit-json --list-out "$_staged_list" \
  | python "$_EVAL_SCRIPTS_DIR/fixed_entry.py" \
      --module RAIN/opt_diagonal_net/custom_optimizer.py \
      --inputs-json-stdin \
      --inputs-glob "RAIN/opt_diagonal_net/_inputs/d200_k20_a0p001_*.npz.b64" \
      --inject-module fixed_benchmark \
      --entry main \
      -- \
      --seed "${SEED:-42}" \
      --label "${ENV:-d200_k20_a1e3}" \
      --output-dir "$OUT_DIR" \
      --dim 200 \
      --sparsity 20 \
      --delta 0.5 \
      --alpha-init 0.001 \
      "${EXTRA_ARGS[@]}"
