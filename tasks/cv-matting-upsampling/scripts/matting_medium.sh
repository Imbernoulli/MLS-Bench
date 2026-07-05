#!/bin/bash
# cv-matting-upsampling: train the matting net with the agent's 'upsampling' surface, then score alpha SAD in
# the trimap UNKNOWN band on the 'medium' trimap-width val setting.
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/image-matting
python harness.py \
    --data-root ${MATTING_DATA_ROOT:-/data/image-matting/composites} \
    --surface upsampling \
    --trimap-width medium \
    --label medium \
    --solution solution/upsampling.py \
    --iters 400 \
    --seed ${SEED:-42}
