#!/bin/bash
# DEPRECATED hidden-seed script -- not referenced by config.json. The scored settings are
# ms/mm/ml (deblur_ms.sh / deblur_mm.sh / deblur_ml.sh). Kept as a harmless seed-1 run of
# the primary (mm) setting if invoked directly.
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/image-deblur
python harness.py \
    --data-root ${DEBLUR_DATA_ROOT:-/data/image-deblur} \
    --surface multiscale \
    --blur-type mm \
    --label mm \
    --solution solution/multiscale.py \
    --iters ${DEBLUR_ITERS:-1500} \
    --seed 1
