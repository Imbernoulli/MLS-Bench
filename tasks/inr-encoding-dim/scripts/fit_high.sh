#!/bin/bash
# inr-encoding-dim (HIGH-frequency signal): fit a fixed coordinate MLP to the high target with the
# agent's chosen design surface (Positional-encoding DIMENSION (number of random Fourier features) at fixed sigma.), then score full-grid reconstruction PSNR.
set -euo pipefail
cd /workspace/inr-signal-fitting

: "${MLSBENCH_VERIFIER_DATA_ROOT:?MLSBENCH_VERIFIER_DATA_ROOT is required}"
export INR_DATA="${MLSBENCH_VERIFIER_DATA_ROOT}/inr-signal-fitting"

python harness.py \
    --solution solution/encoding_dim.py \
    --signal high \
    --seed "${SEED:-0}" \
    --label high
