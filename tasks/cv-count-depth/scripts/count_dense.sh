#!/bin/bash
# cv-count-depth (dense scene): train the agent surface on the DENSE crowd-density
# scene, score counting MAE on its higher-count val split.
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/crowd-counting
python harness.py \
    --data-root ${COUNT_DATA_ROOT:-/data/crowd-counting}/dense \
    --surface depth \
    --label dense \
    --solution crowd-counting/solution/depth.py \
    --iters ${COUNT_ITERS:-450} \
    --seed ${SEED:-42}
