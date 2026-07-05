#!/bin/bash
# cv-count-formulation (medium scene): train fixed frontend + agent count head on the
# MEDIUM crowd-density scene, score counting MAE on its higher-count val split.
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/crowd-counting
python harness.py \
    --data-root ${COUNT_DATA_ROOT:-/data/crowd-counting}/medium \
    --surface head \
    --label medium \
    --solution solution/head.py \
    --iters ${COUNT_ITERS:-1500} \
    --seed ${SEED:-42}
