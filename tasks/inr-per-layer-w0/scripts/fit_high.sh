#!/bin/bash
# inr-per-layer-w0 (HIGH-frequency signal): fit a fixed coordinate MLP to the high target with the
# agent's chosen design surface (SIREN first-layer frequency w0 (too small vs tuned ~30 vs the concave peak).), then score full-grid reconstruction PSNR.
set -euo pipefail
cd /workspace/inr-signal-fitting

: "${MLSBENCH_VERIFIER_DATA_ROOT:?MLSBENCH_VERIFIER_DATA_ROOT is required}"
export INR_DATA="${MLSBENCH_VERIFIER_DATA_ROOT}/inr-signal-fitting"

python harness.py \
    --solution solution/per_layer_w0.py \
    --signal high \
    --seed "${SEED:-0}" \
    --label high
