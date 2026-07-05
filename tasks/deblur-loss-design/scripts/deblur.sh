#!/bin/bash
# DEPRECATED single-setting script -- superseded by deblur_small.sh / deblur_medium.sh /
# deblur_large.sh (the three scored settings in config.json). Kept as a harmless default
# that runs the primary (medium) setting if invoked directly.
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/image-deblur
python harness.py \
    --data-root ${DEBLUR_DATA_ROOT:-/data/image-deblur} \
    --surface loss \
    --blur-type medium \
    --label medium \
    --solution solution/loss.py \
    --iters ${DEBLUR_ITERS:-1500} \
    --seed ${SEED:-42}
