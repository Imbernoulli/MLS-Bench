#!/bin/bash
# gp-likelihood-noise: train an ExactGP (fixed ARD-Matern52 covar + ConstantMean) on
# the FIXED concrete split with the agent's LIKELIHOOD / noise model
# (solution/likelihood_noise.py), then score test NLL (original y scale) + RMSE.
set -euo pipefail
trap 'rc=$?; if [ "$rc" -ne 0 ]; then printf "VERIFICATION_FAILED task=gp-likelihood-noise rc=%s\\n" "$rc"; fi' EXIT
cd "${GP_WORKSPACE:-/workspace/gpytorch-gp}"
export GP_DATA="${MLSBENCH_TASK_DIR:-${TASK_DIR:-/tests/meta}}/data/gpytorch-gp"

python harness_exact.py \
    --task gp-likelihood-noise \
    --dataset concrete \
    --solution solution/likelihood_noise.py \
    --surface likelihood_noise \
    --iters 200 \
    --lr 0.1 \
    --seed "${SEED:-42}"
