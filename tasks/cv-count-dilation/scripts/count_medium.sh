#!/bin/bash
# cv-count-dilation (medium scene): train the agent surface on the MEDIUM crowd-density
# scene, score counting MAE on its higher-count val split.
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/crowd-counting
python harness.py \
    --data-root ${COUNT_DATA_ROOT:-/data/crowd-counting}/medium \
    --surface dilation \
    --label medium \
    --solution crowd-counting/solution/dilation.py \
    --iters ${COUNT_ITERS:-450} \
    --seed ${SEED:-42}
