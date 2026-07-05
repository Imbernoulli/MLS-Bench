#!/bin/bash
# mono3d-head-capacity [moderate]: train the fixed mono-3D model with the agent's surface
# (solution/backbone.py), then score AP3D / 3D-IoU on the TEST objects in the
# 'moderate' tier. Training is on the full fixed train split; only scoring is sliced.
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/mono3d-detection

python harness.py \
    --task backbone \
    --solution solution/backbone.py \
    --label moderate \
    --setting moderate \
    --seed ${SEED:-42}
