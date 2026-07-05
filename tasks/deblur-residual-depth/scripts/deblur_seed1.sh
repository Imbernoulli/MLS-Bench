#!/bin/bash
# Alternate-seed variant (primary mm setting, seed 1) for seed-robustness checks.
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/image-deblur
python harness.py \
    --data-root ${DEBLUR_DATA_ROOT:-/data/image-deblur} \
    --surface depth --blur-type mm --label mm \
    --solution solution/arch_depth.py --iters ${DEBLUR_ITERS:-1500} --seed 1
