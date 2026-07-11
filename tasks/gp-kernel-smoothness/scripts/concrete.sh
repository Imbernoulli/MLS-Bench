#!/bin/bash
# gp-kernel-smoothness: train an ExactGP on the FIXED concrete split with the agent's covariance design
# (solution/kernel_smoothness.py), then score held-out test NLL (original y scale, lower better)+RMSE.
set -euo pipefail
trap 'rc=$?; if [ "$rc" -ne 0 ]; then printf "VERIFICATION_FAILED task=gp-kernel-smoothness rc=%s\\n" "$rc"; fi' EXIT
cd "${GP_WORKSPACE:-/workspace/gpytorch-gp}"
export GP_DATA="${MLSBENCH_TASK_DIR:-${TASK_DIR:-/tests/meta}}/data/gpytorch-gp"

python harness_kernel.py \
    --task gp-kernel-smoothness \
    --dataset concrete \
    --solution solution/kernel_smoothness.py \
    --surface smoothness \
    --iters 200 \
    --lr 0.1 \
    --seed "${SEED:-42}"
