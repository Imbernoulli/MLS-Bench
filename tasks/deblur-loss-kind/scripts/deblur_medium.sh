#!/bin/bash
# deblur-loss-kind [medium = motion-blur severity medium]: train the fixed deblur net with the agent-designed
# loss configuration, score deblur PSNR on the held-out val split.
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/image-deblur
python harness.py \
    --data-root ${DEBLUR_DATA_ROOT:-/data/image-deblur} \
    --surface loss \
    --blur-type medium \
    --label medium \
    --solution solution/losskind.py \
    --iters ${DEBLUR_ITERS:-1500} \
    --seed ${SEED:-42}
