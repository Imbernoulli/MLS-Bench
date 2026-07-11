#!/bin/bash
# gp-svgp-lr: train an SVGP (fixed k-means M=256 inducing, Cholesky, batch 1024) on
# the FIXED concrete split with the agent's Adam LEARNING RATE (solution/svgp_lr.py), then
# score held-out test NLL (original y scale, lower better) + RMSE.
set -euo pipefail
trap 'rc=$?; if [ "$rc" -ne 0 ]; then printf "VERIFICATION_FAILED task=gp-svgp-lr rc=%s\\n" "$rc"; fi' EXIT
cd "${GP_WORKSPACE:-/workspace/gpytorch-gp}"
export GP_DATA="${MLSBENCH_TASK_DIR:-${TASK_DIR:-/tests/meta}}/data/gpytorch-gp"

python harness_svgp.py \
    --task gp-svgp-lr \
    --dataset concrete \
    --solution solution/svgp_lr.py \
    --surface svgp_lr \
    --epochs 20 \
    --batch-size 1024 \
    --fixed-inducing 256 \
    --seed "${SEED:-42}"
