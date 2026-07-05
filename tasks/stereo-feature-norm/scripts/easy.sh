#!/bin/bash
# stereo-feature-norm: train the FIXED small GC-Net/PSMNet-style stereo net with the agent's
# design (surface `featnorm`) on the easy difficulty setting (disparity
# range), then report validation EPE (LOWER is better).
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/stereo-matching

python harness.py \
    --task featnorm \
    --severity easy \
    --solution solution/featnorm.py \
    --label easy \
    --seed ${SEED:-42} \
    --steps ${STEREO_STEPS:-1200}
