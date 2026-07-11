#!/bin/bash
# gp-kernel-design: train an ExactGP on the FIXED elevators regression split with the
# agent's covariance + mean design (solution/kernel_design.py), then score held-out
# test NLL (per point, original y scale, lower is better) + RMSE.
set -euo pipefail
trap 'rc=$?; if [ "$rc" -ne 0 ]; then printf "VERIFICATION_FAILED task=gp-kernel-design rc=%s\\n" "$rc"; fi' EXIT
# Settings are assigned distinct verifier groups and run serially on one GPU.
# Inherit the GPU selected by the runner.
cd "${GP_WORKSPACE:-/workspace/gpytorch-gp}"
export GP_DATA="${MLSBENCH_TASK_DIR:-${TASK_DIR:-/tests/meta}}/data/gpytorch-gp"

python harness_kernel.py \
    --task gp-kernel-design \
    --surface kernel_design \
    --dataset elevators \
    --solution solution/kernel_design.py \
    --iters 200 \
    --lr 0.1 \
    --seed "${SEED:-42}"
