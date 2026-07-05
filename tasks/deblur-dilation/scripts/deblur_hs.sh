#!/bin/bash
# deblur-dilation [hs = motion-blur severity hs]: train the fixed deblur net with the agent-designed
# dilation configuration, score deblur PSNR on the held-out val split.
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/image-deblur
python harness.py \
    --data-root ${DEBLUR_DATA_ROOT:-/data/image-deblur} \
    --surface dilation \
    --blur-type hs \
    --label hs \
    --solution solution/arch_dilation.py \
    --iters ${DEBLUR_ITERS:-1500} \
    --seed ${SEED:-42}
