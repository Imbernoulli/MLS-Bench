#!/bin/bash
set -euo pipefail

cd /workspace

OUT_DIR="${OUTPUT_DIR:-${SAVE_PATH:-/tmp/mlsbench_optimization_parity}}"

python pytorch-examples/optimization_parity/custom_strategy.py   --seed "${SEED:-42}"   --label "${ENV:-n16-k4}"   --output-dir "$OUT_DIR"   --n-features 16   --secret-size 4
