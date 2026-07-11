#!/bin/bash
# inr-eikonal-reg (MEDIUM-frequency signal): vary the per-channel RGB Jacobian
# smoothness weight, then score full-grid reconstruction PSNR.
set -euo pipefail
cd /workspace/inr-signal-fitting

: "${MLSBENCH_VERIFIER_DATA_ROOT:?MLSBENCH_VERIFIER_DATA_ROOT is required}"
export INR_DATA="${MLSBENCH_VERIFIER_DATA_ROOT}/inr-signal-fitting"

python harness.py \
    --solution solution/jacobian_reg.py \
    --signal medium \
    --seed "${SEED:-0}" \
    --label medium
