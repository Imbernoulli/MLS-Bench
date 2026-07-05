#!/bin/bash
# DEPRECATED single-setting script -- superseded by the three scored settings in config.json.
# Kept as a harmless default that runs the primary (mm) setting if invoked directly.
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/image-deblur
python harness.py \
    --data-root ${DEBLUR_DATA_ROOT:-/data/image-deblur} \
    --surface attention --blur-type mm --label mm \
    --solution solution/arch_attention.py --iters ${DEBLUR_ITERS:-1500} --seed ${SEED:-42}
