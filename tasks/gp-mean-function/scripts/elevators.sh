#!/bin/bash
# gp-mean-function: train an ExactGP (fixed ARD-Matern52 covar + GaussianLikelihood)
# on the FIXED elevators split with the agent's MEAN function (solution/mean_function.py),
# then score held-out test NLL (original y scale, lower better) + RMSE.
set -euo pipefail
trap 'rc=$?; if [ "$rc" -ne 0 ]; then printf "VERIFICATION_FAILED task=gp-mean-function rc=%s\\n" "$rc"; fi' EXIT
cd "${GP_WORKSPACE:-/workspace/gpytorch-gp}"
export GP_DATA="${MLSBENCH_TASK_DIR:-${TASK_DIR:-/tests/meta}}/data/gpytorch-gp"

python harness_exact.py \
    --task gp-mean-function \
    --dataset elevators \
    --solution solution/mean_function.py \
    --surface mean_function \
    --iters 200 \
    --lr 0.1 \
    --seed "${SEED:-42}"
