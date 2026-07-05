#!/bin/bash
# cv-count-normalization (middense scene): train fixed frontend + agent density head on
# the MIDDENSE crowd-density scene, score counting MAE on its higher-count val split.
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/crowd-counting
python harness.py \
    --data-root ${COUNT_DATA_ROOT:-/data/crowd-counting}/middense \
    --surface norm \
    --label middense \
    --solution solution/norm.py \
    --iters ${COUNT_ITERS:-1500} \
    --seed ${SEED:-42}
