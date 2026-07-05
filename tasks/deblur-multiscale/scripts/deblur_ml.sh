#!/bin/bash
# deblur-multiscale [ml = heavy motion blur, largest severity of the heavy band]: train the
# fixed scale-recurrent deblur net with the agent-designed number of coarse-to-fine scales,
# score deblur PSNR on the held-out val split.
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/image-deblur
python harness.py \
    --data-root ${DEBLUR_DATA_ROOT:-/data/image-deblur} \
    --surface multiscale \
    --blur-type ml \
    --label ml \
    --solution solution/multiscale.py \
    --iters ${DEBLUR_ITERS:-1500} \
    --seed ${SEED:-42}
