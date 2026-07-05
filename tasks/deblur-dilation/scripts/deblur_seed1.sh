#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/image-deblur
python harness.py --data-root ${DEBLUR_DATA_ROOT:-/data/image-deblur} --surface dilation --blur-type hm --label hm --solution solution/arch_dilation.py --iters ${DEBLUR_ITERS:-1500} --seed 1
