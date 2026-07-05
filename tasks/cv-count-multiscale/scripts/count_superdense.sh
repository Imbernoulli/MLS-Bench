#!/bin/bash
# cv-count-multiscale (superdense scene): train the agent surface on the SUPERDENSE crowd-density
# scene, score counting MAE on its higher-count val split.
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/crowd-counting
python harness.py \
    --data-root ${COUNT_DATA_ROOT:-/data/crowd-counting}/superdense \
    --surface multiscale \
    --label superdense \
    --solution crowd-counting/solution/multiscale.py \
    --iters ${COUNT_ITERS:-450} \
    --seed ${SEED:-42}
