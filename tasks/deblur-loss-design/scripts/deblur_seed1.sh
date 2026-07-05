#!/bin/bash
# DEPRECATED hidden-seed script -- not referenced by config.json. The scored settings are
# small/medium/large (deblur_small.sh / deblur_medium.sh / deblur_large.sh). Kept as a
# harmless seed-1 run of the primary (medium) setting if invoked directly.
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
    --seed 1
