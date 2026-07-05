#!/bin/bash
# deblur-edge-loss [rl = motion-blur severity rl]: train the fixed deblur net with the agent-designed
# edge/gradient-loss configuration, score deblur PSNR on the held-out val split.
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/image-deblur
python harness.py \
    --data-root ${DEBLUR_DATA_ROOT:-/data/image-deblur} \
    --surface edge \
    --blur-type rl \
    --label rl \
    --solution solution/edge.py \
    --iters ${DEBLUR_ITERS:-1500} \
    --seed ${SEED:-42}
