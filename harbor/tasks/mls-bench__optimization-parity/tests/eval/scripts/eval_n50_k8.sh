#!/bin/bash
set -euo pipefail

cd /workspace

OUT_DIR="${OUTPUT_DIR:-${SAVE_PATH:-/tmp/mlsbench_optimization_parity}}"

python pytorch-examples/optimization_parity/custom_strategy.py   --seed "${SEED:-42}"   --label "${ENV:-n50-k8}"   --output-dir "$OUT_DIR"   --n-features 50   --secret-size 8
