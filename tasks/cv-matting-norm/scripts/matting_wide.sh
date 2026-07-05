#!/bin/bash
# cv-matting-norm: train the matting net with the agent's 'norm' surface, then score alpha SAD in
# the trimap UNKNOWN band on the 'wide' trimap-width val setting.
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/image-matting
python harness.py \
    --data-root ${MATTING_DATA_ROOT:-/data/image-matting/composites} \
    --surface norm \
    --trimap-width wide \
    --label wide \
    --solution solution/norm.py \
    --iters 400 \
    --seed ${SEED:-42}
