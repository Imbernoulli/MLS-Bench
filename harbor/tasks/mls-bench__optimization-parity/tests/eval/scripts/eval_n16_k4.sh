#!/bin/bash
cd /workspace

# Inputs for this run are re-materialized by apply.py below, so the runner may
# treat them as ephemeral (load into memory, then delete before editable code
# runs).
export MLSBENCH_EPHEMERAL_INPUTS=1
# Splice the fixed runner blocks into stale baked workspace images (anchored
# exact-match replaces; no-op when the workspace is already up to date).
source "$(dirname "${BASH_SOURCE[0]}")/_runtime_patch.sh"
python "/tests/eval/_inputgen/apply.py" "optimization-parity" /workspace
set -euo pipefail
OUT_DIR="${OUTPUT_DIR:-${SAVE_PATH:-/tmp/mlsbench_optimization_parity}}"
python pytorch-examples/optimization_parity/custom_strategy.py   --seed "${SEED:-42}"   --label "${ENV:-n16-k4}"   --output-dir "$OUT_DIR"   --n-features 16   --secret-size 4
