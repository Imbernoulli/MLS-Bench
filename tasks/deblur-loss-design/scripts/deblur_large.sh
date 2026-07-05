#!/bin/bash
# deblur-loss-design [large motion blur]: train the fixed residual deblur net with the
# agent-designed reconstruction target/loss, score deblur PSNR on the held-out val split.
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/image-deblur
python harness.py \
    --data-root ${DEBLUR_DATA_ROOT:-/data/image-deblur} \
    --surface loss \
    --blur-type large \
    --label large \
    --solution solution/loss.py \
    --iters ${DEBLUR_ITERS:-1500} \
    --seed ${SEED:-42}
