#!/bin/bash
# cv-matting-decoder-design: train the matting net with the agent's 'decoder' surface, then score alpha SAD in
# the trimap UNKNOWN band on the 'xwide' trimap-width val setting.
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/image-matting
python harness.py \
    --data-root ${MATTING_DATA_ROOT:-/data/image-matting/composites} \
    --surface decoder \
    --trimap-width xwide \
    --label xwide \
    --solution solution/decoder.py \
    --iters 400 \
    --seed ${SEED:-42}
