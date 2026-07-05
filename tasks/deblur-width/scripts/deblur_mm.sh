#!/bin/bash
# deblur-width [mm = motion-blur severity mm]: train the fixed deblur net with the agent-designed
# width configuration, score deblur PSNR on the held-out val split.
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/image-deblur
python harness.py \
    --data-root ${DEBLUR_DATA_ROOT:-/data/image-deblur} \
    --surface width \
    --blur-type mm \
    --label mm \
    --solution solution/arch_width.py \
    --iters ${DEBLUR_ITERS:-1500} \
    --seed ${SEED:-42}
