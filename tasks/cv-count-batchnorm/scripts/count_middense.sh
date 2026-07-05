#!/bin/bash
# cv-count-batchnorm (middense scene): train the agent surface on the MIDDENSE crowd-density
# scene, score counting MAE on its higher-count val split.
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/crowd-counting
python harness.py \
    --data-root ${COUNT_DATA_ROOT:-/data/crowd-counting}/middense \
    --surface batchnorm \
    --label middense \
    --solution crowd-counting/solution/batchnorm.py \
    --iters ${COUNT_ITERS:-450} \
    --seed ${SEED:-42}
