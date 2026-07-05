#!/bin/bash
# mono3d-height-source [easy]: train the fixed mono-3D model with the agent's surface
# (solution/height_source.py), then score AP3D / 3D-IoU on the TEST objects in the
# 'easy' tier. Training is on the full fixed train split; only scoring is sliced.
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/mono3d-detection

python harness.py \
    --task height_source \
    --solution solution/height_source.py \
    --label easy \
    --setting easy \
    --seed ${SEED:-42}
