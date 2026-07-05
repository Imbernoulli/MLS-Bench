#!/bin/bash
# cv-count-formulation (dense scene, hidden): train fixed frontend + agent count head on
# the DENSE crowd-density scene, score counting MAE on its higher-count val split.
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/crowd-counting
python harness.py \
    --data-root ${COUNT_DATA_ROOT:-/data/crowd-counting}/dense \
    --surface head \
    --label dense \
    --solution solution/head.py \
    --iters ${COUNT_ITERS:-1500} \
    --seed ${SEED:-42}
