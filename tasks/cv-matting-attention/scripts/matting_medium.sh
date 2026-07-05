#!/bin/bash
# cv-matting-attention: train the matting net with the agent's 'attention' surface, then score alpha SAD in
# the trimap UNKNOWN band on the 'medium' trimap-width val setting.
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/image-matting
python harness.py \
    --data-root ${MATTING_DATA_ROOT:-/data/image-matting/composites} \
    --surface attention \
    --trimap-width medium \
    --label medium \
    --solution solution/attention.py \
    --iters 400 \
    --seed ${SEED:-42}
