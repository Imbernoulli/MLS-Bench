#!/bin/bash
# stereo-agg-normalization: train the FIXED small GC-Net/PSMNet-style stereo net with the agent's
# design (surface `normalization`) on the medium difficulty setting (disparity
# range), then report validation EPE (LOWER is better).
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/stereo-matching

python harness.py \
    --task normalization \
    --severity medium \
    --solution solution/normalization.py \
    --label medium \
    --seed ${SEED:-42} \
    --steps ${STEREO_STEPS:-1200}
