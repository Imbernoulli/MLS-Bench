#!/bin/bash
# cv-harmonization-loss-region: train the harmonizer with the agent-designed loss surface on the
# MEDIUM appearance-mismatch setting, then score FOREGROUND-region PSNR vs the GT.
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/image-harmonization
python harness.py \
    --data-root ${HARMONY_DATA_ROOT:-/data/image-harmonization} \
    --surface loss \
    --severity medium \
    --label medium \
    --solution solution/loss.py \
    --iters ${HARMONY_ITERS:-500} \
    --seed ${SEED:-42}
