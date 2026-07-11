#!/bin/bash
# inr-hash-grid (MEDIUM-frequency signal): fit a fixed coordinate MLP to the medium target with the
# agent's chosen design surface (Multiresolution HASH-GRID encoding structure (Instant-NGP): collapsed vs proper pyramid.), then score full-grid reconstruction PSNR.
set -euo pipefail
cd /workspace/inr-signal-fitting

: "${MLSBENCH_VERIFIER_DATA_ROOT:?MLSBENCH_VERIFIER_DATA_ROOT is required}"
export INR_DATA="${MLSBENCH_VERIFIER_DATA_ROOT}/inr-signal-fitting"

python harness.py \
    --solution solution/hash_grid.py \
    --signal medium \
    --seed "${SEED:-0}" \
    --label medium
