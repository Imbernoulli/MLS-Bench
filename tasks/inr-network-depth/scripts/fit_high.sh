#!/bin/bash
# inr-network-depth (HIGH-frequency signal): fit a fixed coordinate MLP to the high target with the
# agent's chosen design surface (DEPTH of a Fourier-encoded ReLU coordinate MLP (too shallow vs well-chosen vs too deep).), then score full-grid reconstruction PSNR.
set -euo pipefail
cd /workspace/inr-signal-fitting

: "${MLSBENCH_VERIFIER_DATA_ROOT:?MLSBENCH_VERIFIER_DATA_ROOT is required}"
export INR_DATA="${MLSBENCH_VERIFIER_DATA_ROOT}/inr-signal-fitting"

python harness.py \
    --solution solution/depth.py \
    --signal high \
    --seed "${SEED:-0}" \
    --label high
