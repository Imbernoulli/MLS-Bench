#!/bin/bash
# gp-sparse-inducing: train a Stochastic Variational GP (SVGP) on the FIXED concrete
# regression split with the agent's inducing-point selection (solution/inducing.py),
# then score held-out test NLL (original y scale, lower is better) + RMSE.
set -euo pipefail
trap 'rc=$?; if [ "$rc" -ne 0 ]; then printf "VERIFICATION_FAILED task=gp-sparse-inducing rc=%s\\n" "$rc"; fi' EXIT
cd "${GP_WORKSPACE:-/workspace/gpytorch-gp}"
export GP_DATA="${MLSBENCH_TASK_DIR:-${TASK_DIR:-/tests/meta}}/data/gpytorch-gp"

python harness_sparse.py \
    --task gp-sparse-inducing \
    --surface inducing \
    --dataset concrete \
    --solution solution/inducing.py \
    --epochs 20 \
    --batch-size 1024 \
    --lr 0.01 \
    --max-inducing 2048 \
    --seed "${SEED:-42}"
