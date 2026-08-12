#!/bin/bash
cd /workspace

# Inputs for this run are re-materialized by apply.py below, so the runner may
# treat them as ephemeral (load into memory, then delete before editable code
# runs).
export MLSBENCH_EPHEMERAL_INPUTS=1
# Splice the fixed runner blocks into stale baked workspace images (anchored
# exact-match replaces; no-op when the workspace is already up to date).
source "$(dirname "${BASH_SOURCE[0]}")/_runtime_patch.sh"
_staged_list="$(mktemp /tmp/mlsb_staged.XXXXXX)"
# EXIT backstop: delete exactly the blobs apply.py staged for THIS eval,
# even when the runner crashes/times out before its in-process scrub —
# a crashed eval must not leave its withheld inputs readable for the rest
# of the wave. Normally a no-op (the runner scrubs right after loading).
trap 'xargs -r rm -f -- < "$_staged_list" 2>/dev/null; rm -f "$_staged_list"' EXIT
python "/tests/eval/_inputgen/apply.py" "optimization-parity" /workspace --list-out "$_staged_list"
set -euo pipefail
OUT_DIR="${OUTPUT_DIR:-${SAVE_PATH:-/tmp/mlsbench_optimization_parity}}"
python pytorch-examples/optimization_parity/custom_strategy.py   --seed "${SEED:-42}"   --label "${ENV:-eval}"   --output-dir "$OUT_DIR"
