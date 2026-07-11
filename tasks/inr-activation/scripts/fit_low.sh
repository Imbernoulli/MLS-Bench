#!/bin/bash
# inr-activation (LOW-frequency signal): fit a fixed-capacity coordinate MLP to the REAL Kodak LOW-frequency target (kodim10, calm dockside scene) with the agent's chosen ACTIVATION+encoding (plain ReLU vs
set -euo pipefail
# Settings use distinct execution groups and run serially. Inherit the verifier-assigned CUDA_VISIBLE_DEVICES value.
cd /workspace/inr-signal-fitting

: "${MLSBENCH_VERIFIER_DATA_ROOT:?MLSBENCH_VERIFIER_DATA_ROOT is required}"
export INR_DATA="${MLSBENCH_VERIFIER_DATA_ROOT}/inr-signal-fitting"

python harness.py \
    --solution solution/activation.py \
    --signal low \
    --seed "${SEED:-0}" \
    --label low
