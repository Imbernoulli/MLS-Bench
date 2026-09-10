#!/bin/bash
set -euo pipefail

# Resolve the FIXED wrapper next to this script BEFORE any cd (robust to a
# relative invocation path).
FIXED_ENTRY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/fixed_entry.py"

cd /workspace

# The harness re-stages this run's pre-generated input blobs before every
# evaluation, so they are treated as ephemeral (read+unlinked by the fixed
# wrapper before any editable code runs).
export MLSBENCH_EPHEMERAL_INPUTS=1

OUT_DIR="${OUTPUT_DIR:-${SAVE_PATH:-/tmp/mlsbench_optimization_parity}}"

# FIXED wrapper: preloads and unlinks this run's staged input blobs BEFORE
# importing the editable module, so top-level statements in the editable
# range never see them on disk.
python "$FIXED_ENTRY" \
  --module pytorch-examples/optimization_parity/custom_strategy.py \
  --inputs-glob "pytorch-examples/optimization_parity/_parity_inputs/n64_k8_seed${SEED:-42}_s*.labels.b64" \
  --entry main \
  -- --seed "${SEED:-42}"   --label "${ENV:-n64-k8}"   --output-dir "$OUT_DIR"   --n-features 64   --secret-size 8
