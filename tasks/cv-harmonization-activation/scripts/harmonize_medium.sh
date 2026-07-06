#!/bin/bash
# cv-harmonization-activation: train the harmonizer with the agent-designed activation surface on the
# MEDIUM appearance-mismatch setting, then score FOREGROUND-region PSNR vs the GT.
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/image-harmonization
python harness.py \
    --data-root ${HARMONY_DATA_ROOT:-/data/image-harmonization} \
    --surface activation \
    --severity medium \
    --label medium \
    --solution solution/activation.py \
    --iters ${HARMONY_ITERS:-500} \
    --seed ${SEED:-42}
