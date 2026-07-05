#!/bin/bash
# deshadow-mask-guidance (setting: heavy cast shadow): train the residual deshadower with the
# agent-designed backbone, then score SHADOW-REGION PSNR on the held-out val split.
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/image-deshadow
python harness.py \
    --data-root ${DESHADOW_DATA_ROOT:-/data/image-deshadow}/heavy \
    --surface network \
    --label heavy \
    --solution solution/network.py \
    --iters ${DESHADOW_ITERS:-400} \
    --seed ${SEED:-42}
