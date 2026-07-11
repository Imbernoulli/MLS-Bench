#!/bin/bash
# gp-deep-kernel: train a Deep Kernel Learning GP on the FIXED Kin8nm regression
# split (8192x8) with the agent's NN feature extractor (solution/deep_kernel.py)
# feeding a fixed ScaleKernel(RBFKernel(ard)) GP head trained jointly under the
# exact marginal log-likelihood, then score held-out test NLL (original y scale,
# lower is better) + RMSE.
set -euo pipefail
trap 'rc=$?; if [ "$rc" -ne 0 ]; then printf "VERIFICATION_FAILED task=gp-deep-kernel rc=%s\\n" "$rc"; fi' EXIT
cd "${GP_WORKSPACE:-/workspace/gpytorch-gp}"
export GP_DATA="${MLSBENCH_TASK_DIR:-${TASK_DIR:-/tests/meta}}/data/gpytorch-gp"

python harness_deepkernel.py \
    --task gp-deep-kernel \
    --surface deep_kernel \
    --dataset kin8nm \
    --solution solution/deep_kernel.py \
    --iters 200 \
    --lr 0.01 \
    --seed "${SEED:-42}"
