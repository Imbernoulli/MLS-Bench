#!/bin/bash
# cv-matting-loss-design: train the matting net with the agent's 'loss' surface, then score alpha SAD in
# the trimap UNKNOWN band on the 'medium' trimap-width val setting.
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/image-matting
python harness.py \
    --data-root ${MATTING_DATA_ROOT:-/data/image-matting/composites} \
    --surface loss \
    --trimap-width medium \
    --label medium \
    --solution solution/loss.py \
    --iters 250 \
    --seed ${SEED:-42}
