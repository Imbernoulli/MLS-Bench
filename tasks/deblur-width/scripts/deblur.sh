#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/image-deblur
python harness.py --data-root ${DEBLUR_DATA_ROOT:-/data/image-deblur} --surface width --blur-type mm --label mm --solution solution/arch_width.py --iters ${DEBLUR_ITERS:-1500} --seed ${SEED:-42}
