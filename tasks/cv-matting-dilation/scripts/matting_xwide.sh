#!/bin/bash
# cv-matting-dilation: train the matting net with the agent's 'dilation' surface, then score alpha SAD in
# the trimap UNKNOWN band on the 'xwide' trimap-width val setting.
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/image-matting
python harness.py \
    --data-root ${MATTING_DATA_ROOT:-/data/image-matting/composites} \
    --surface dilation \
    --trimap-width xwide \
    --label xwide \
    --solution solution/dilation.py \
    --iters 400 \
    --seed ${SEED:-42}
