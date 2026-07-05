#!/bin/bash
# cv-matting-skip: train the matting net with the agent's 'skip' surface, then score alpha SAD in
# the trimap UNKNOWN band on the 'medium' trimap-width val setting.
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/image-matting
python harness.py \
    --data-root ${MATTING_DATA_ROOT:-/data/image-matting/composites} \
    --surface skip \
    --trimap-width medium \
    --label medium \
    --solution solution/skip.py \
    --iters 400 \
    --seed ${SEED:-42}
