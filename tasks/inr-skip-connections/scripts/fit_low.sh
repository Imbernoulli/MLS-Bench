#!/bin/bash
# inr-skip-connections (LOW-frequency signal): fit a fixed coordinate MLP to the low target with the
# agent's chosen design surface (NeRF-style SKIP CONNECTION in a deep (8-layer) Fourier-encoded MLP (no-skip vs skip).), then score full-grid reconstruction PSNR.
set -euo pipefail
cd /workspace/inr-signal-fitting

: "${MLSBENCH_VERIFIER_DATA_ROOT:?MLSBENCH_VERIFIER_DATA_ROOT is required}"
export INR_DATA="${MLSBENCH_VERIFIER_DATA_ROOT}/inr-signal-fitting"

python harness.py \
    --solution solution/skip.py \
    --signal low \
    --seed "${SEED:-0}" \
    --label low
