#!/bin/bash
# cv-harmonization-input-norm: train the harmonizer with the agent-designed inputnorm surface on the
# STRONG appearance-mismatch setting, then score FOREGROUND-region PSNR vs the GT.
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/image-harmonization
python harness.py \
    --data-root ${HARMONY_DATA_ROOT:-/data/image-harmonization} \
    --surface inputnorm \
    --severity strong \
    --label strong \
    --solution solution/inputnorm.py \
    --iters ${HARMONY_ITERS:-500} \
    --seed ${SEED:-42}
