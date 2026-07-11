#!/bin/bash
# gp-ard-lengthscale: train an ExactGP on the FIXED concrete split with the agent's covariance design
# (solution/ard_lengthscale.py), then score held-out test NLL (original y scale, lower better)+RMSE.
set -euo pipefail
trap 'rc=$?; if [ "$rc" -ne 0 ]; then printf "VERIFICATION_FAILED task=gp-ard-lengthscale rc=%s\\n" "$rc"; fi' EXIT
cd "${GP_WORKSPACE:-/workspace/gpytorch-gp}"
export GP_DATA="${MLSBENCH_TASK_DIR:-${TASK_DIR:-/tests/meta}}/data/gpytorch-gp"

python harness_kernel.py \
    --task gp-ard-lengthscale \
    --dataset concrete \
    --solution solution/ard_lengthscale.py \
    --surface ard \
    --iters 200 \
    --lr 0.1 \
    --seed "${SEED:-42}"
