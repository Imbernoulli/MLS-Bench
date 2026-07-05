#!/bin/bash
# deblur-recurrence [ml = motion-blur severity ml]: train the fixed deblur net with the agent-designed
# recurrence configuration, score deblur PSNR on the held-out val split.
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/image-deblur
python harness.py \
    --data-root ${DEBLUR_DATA_ROOT:-/data/image-deblur} \
    --surface recurrence \
    --blur-type ml \
    --label ml \
    --solution solution/recurrence.py \
    --iters ${DEBLUR_ITERS:-1500} \
    --seed ${SEED:-42}
