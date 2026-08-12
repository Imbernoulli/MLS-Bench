#!/bin/bash
set -euo pipefail

cd /workspace

# The harness re-stages this run's pre-generated input blobs before every
# evaluation, so the runner treats them as ephemeral (deleted right after
# loading, before any editable code runs).
export MLSBENCH_EPHEMERAL_INPUTS=1

OUT_DIR="${OUTPUT_DIR:-${SAVE_PATH:-/tmp/mlsbench_optimization_parity}}"

python pytorch-examples/optimization_parity/custom_strategy.py   --seed "${SEED:-42}"   --label "${ENV:-n64-k8}"   --output-dir "$OUT_DIR"   --n-features 64   --secret-size 8
