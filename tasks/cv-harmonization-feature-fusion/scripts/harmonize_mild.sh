#!/bin/bash
# cv-harmonization-feature-fusion: train the harmonizer with the agent-designed fusion surface on the
# MILD appearance-mismatch setting, then score FOREGROUND-region PSNR vs the GT.
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/image-harmonization
python harness.py \
    --data-root ${HARMONY_DATA_ROOT:-/data/image-harmonization} \
    --surface fusion \
    --severity mild \
    --label mild \
    --solution solution/fusion.py \
    --iters ${HARMONY_ITERS:-500} \
    --seed ${SEED:-42}
