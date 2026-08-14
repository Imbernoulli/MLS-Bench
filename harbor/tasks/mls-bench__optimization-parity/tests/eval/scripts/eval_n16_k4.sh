#!/bin/bash
cd /workspace

# Inputs for this run are PIPED to the fixed wrapper as JSON (apply.py
# --emit-json -> fixed_entry.py --inputs-json-stdin) and NEVER touch the
# shared workspace filesystem: Harbor runs the label x seed wave
# CONCURRENTLY, so an on-disk staging window would be readable by a sibling
# eval's agent code. MLSBENCH_EPHEMERAL_INPUTS=1 marks them ephemeral.
export MLSBENCH_EPHEMERAL_INPUTS=1
_EVAL_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_staged_list="$(mktemp /tmp/mlsb_staged.XXXXXX)"
# EXIT backstop: no-op with --emit-json (nothing is staged to disk); covers
# a version-skewed apply.py that stages the old way — the wrapper's
# --inputs-glob pass unlinks any such stragglers before importing the module.
trap 'xargs -r rm -f -- < "$_staged_list" 2>/dev/null; rm -f "$_staged_list"' EXIT
set -euo pipefail
OUT_DIR="${OUTPUT_DIR:-${SAVE_PATH:-/tmp/mlsbench_optimization_parity}}"

# FIXED wrapper: preloads this run's input blobs (from stdin; the disk glob
# is only an unlink backstop) BEFORE importing the editable module.
python "/tests/eval/_inputgen/apply.py" "optimization-parity" /workspace \
    --emit-json --list-out "$_staged_list" \
  | python "$_EVAL_SCRIPTS_DIR/fixed_entry.py" \
      --module pytorch-examples/optimization_parity/custom_strategy.py \
      --inputs-json-stdin \
      --inputs-glob "pytorch-examples/optimization_parity/_parity_inputs/n16_k4_seed${SEED:-42}_s*.labels.b64" \
      --entry main \
      -- --seed "${SEED:-42}"   --label "${ENV:-n16-k4}"   --output-dir "$OUT_DIR"   --n-features 16   --secret-size 4
