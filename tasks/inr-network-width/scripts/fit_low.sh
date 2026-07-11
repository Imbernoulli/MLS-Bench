#!/bin/bash
# inr-network-width (LOW-frequency signal): fit a fixed coordinate MLP to the low target with the
# agent's chosen design surface (WIDTH (hidden units) of a Fourier-encoded ReLU coordinate MLP (too narrow vs well-chosen).), then score full-grid reconstruction PSNR.
set -euo pipefail
cd /workspace/inr-signal-fitting

: "${MLSBENCH_VERIFIER_DATA_ROOT:?MLSBENCH_VERIFIER_DATA_ROOT is required}"
export INR_DATA="${MLSBENCH_VERIFIER_DATA_ROOT}/inr-signal-fitting"

python harness.py \
    --solution solution/width.py \
    --signal low \
    --seed "${SEED:-0}" \
    --label low
