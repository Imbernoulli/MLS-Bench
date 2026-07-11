#!/bin/bash
# gp-exact-lr: train an ExactGP (fixed ARD-Matern52 covar, ConstantMean, Gaussian
# likelihood) on the FIXED elevators split with the agent's Adam LEARNING RATE
# (solution/exact_lr.py) over a fixed 200-iteration budget, then score held-out test
# NLL (original y scale, lower better) + RMSE.
set -euo pipefail
trap 'rc=$?; if [ "$rc" -ne 0 ]; then printf "VERIFICATION_FAILED task=gp-exact-lr rc=%s\\n" "$rc"; fi' EXIT
cd "${GP_WORKSPACE:-/workspace/gpytorch-gp}"
export GP_DATA="${MLSBENCH_TASK_DIR:-${TASK_DIR:-/tests/meta}}/data/gpytorch-gp"

python harness_exact.py \
    --task gp-exact-lr \
    --dataset elevators \
    --solution solution/exact_lr.py \
    --surface exact_lr \
    --iters 200 \
    --seed "${SEED:-42}"
