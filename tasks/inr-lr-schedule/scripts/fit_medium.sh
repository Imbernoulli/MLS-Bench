#!/bin/bash
# inr-lr-schedule (MEDIUM-frequency signal): fit a fixed coordinate MLP to the medium target with the
# agent's chosen design surface (LEARNING RATE + SCHEDULE of full-batch Adam (too-large constant vs tuned cosine).), then score full-grid reconstruction PSNR.
set -euo pipefail
cd /workspace/inr-signal-fitting

: "${MLSBENCH_VERIFIER_DATA_ROOT:?MLSBENCH_VERIFIER_DATA_ROOT is required}"
export INR_DATA="${MLSBENCH_VERIFIER_DATA_ROOT}/inr-signal-fitting"

python harness.py \
    --solution solution/lr_schedule.py \
    --signal medium \
    --seed "${SEED:-0}" \
    --label medium
