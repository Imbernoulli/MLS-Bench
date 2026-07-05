#!/bin/bash
# deshadow-residual [light cast shadow]: train the FIXED mask-guided residual deshadower
# with the agent-designed RESIDUAL lever, then score SHADOW-REGION PSNR on the held-out
# val split.
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/image-deshadow
python harness.py \
    --data-root ${DESHADOW_DATA_ROOT:-/data/image-deshadow}/light \
    --surface residual \
    --label light \
    --solution solution/residual.py \
    --iters ${DESHADOW_ITERS:-400} \
    --seed ${SEED:-42}
