#!/bin/bash
# gp-deep-kernel-width: train a Deep Kernel Learning GP on the FIXED elevators split with
# the agent's MLP feature extractor (solution/deep_kernel_width.py) whose only free
# choice is the BOTTLENECK WIDTH p, then score held-out test NLL (lower better)+RMSE.
set -euo pipefail
trap 'rc=$?; if [ "$rc" -ne 0 ]; then printf "VERIFICATION_FAILED task=gp-deep-kernel-width rc=%s\\n" "$rc"; fi' EXIT
cd "${GP_WORKSPACE:-/workspace/gpytorch-gp}"
export GP_DATA="${MLSBENCH_TASK_DIR:-${TASK_DIR:-/tests/meta}}/data/gpytorch-gp"

python harness_deepkernel.py \
    --task gp-deep-kernel-width \
    --dataset elevators \
    --solution solution/deep_kernel_width.py \
    --surface deep_kernel_width \
    --iters 200 \
    --lr 0.01 \
    --seed "${SEED:-42}"
