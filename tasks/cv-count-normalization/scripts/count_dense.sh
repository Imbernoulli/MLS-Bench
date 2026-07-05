#!/bin/bash
# cv-count-normalization (dense scene, hidden): train fixed frontend + agent density
# head on the DENSE crowd-density scene, score counting MAE on its higher-count val.
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/crowd-counting
python harness.py \
    --data-root ${COUNT_DATA_ROOT:-/data/crowd-counting}/dense \
    --surface norm \
    --label dense \
    --solution solution/norm.py \
    --iters ${COUNT_ITERS:-1500} \
    --seed ${SEED:-42}
