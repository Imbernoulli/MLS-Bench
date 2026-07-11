#!/bin/bash
# inr-fourier-frequency (MEDIUM-frequency signal): fit a fixed-capacity coordinate MLP to the REAL Kodak MEDIUM-frequency target (kodim07, flowers + window lattice) with the agent's chosen Fourier FREQUENCY sigma, then score PSNR.
set -euo pipefail
cd /workspace/inr-signal-fitting
: "${MLSBENCH_VERIFIER_DATA_ROOT:?MLSBENCH_VERIFIER_DATA_ROOT is required}"
export INR_DATA="${MLSBENCH_VERIFIER_DATA_ROOT}/inr-signal-fitting"

python harness.py \
    --solution solution/frequency.py \
    --signal medium \
    --seed "${SEED:-0}" \
    --label medium
