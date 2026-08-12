#!/bin/bash
cd /workspace

# Inputs for this run are re-materialized by apply.py below, so they are
# treated as ephemeral (read+unlinked by the fixed wrapper before any
# editable code runs).
export MLSBENCH_EPHEMERAL_INPUTS=1
_EVAL_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Splice the fixed runner blocks into stale baked workspace images (anchored
# exact-match replaces; no-op when the workspace is already up to date).
source "$_EVAL_SCRIPTS_DIR/_runtime_patch.sh"
_staged_list="$(mktemp /tmp/mlsb_staged.XXXXXX)"
# EXIT backstop: delete exactly the blobs apply.py staged for THIS eval,
# even when the runner crashes/times out before its in-process scrub —
# a crashed eval must not leave its withheld inputs readable for the rest
# of the wave. Normally a no-op (the fixed wrapper unlinks right after
# preloading).
trap 'xargs -r rm -f -- < "$_staged_list" 2>/dev/null; rm -f "$_staged_list"' EXIT
python "/tests/eval/_inputgen/apply.py" "optimization-parity" /workspace --list-out "$_staged_list"
set -euo pipefail
OUT_DIR="${OUTPUT_DIR:-${SAVE_PATH:-/tmp/mlsbench_optimization_parity}}"

# FIXED wrapper: preloads and unlinks this run's staged input blobs BEFORE
# importing the editable module, so top-level statements in the editable
# range never see them on disk.
python "$_EVAL_SCRIPTS_DIR/fixed_entry.py" \
  --module pytorch-examples/optimization_parity/custom_strategy.py \
  --inputs-glob "pytorch-examples/optimization_parity/_parity_inputs/n16_k4_seed${SEED:-42}_s*.labels.b64" \
  --entry main \
  -- --seed "${SEED:-42}"   --label "${ENV:-n16-k4}"   --output-dir "$OUT_DIR"   --n-features 16   --secret-size 4
