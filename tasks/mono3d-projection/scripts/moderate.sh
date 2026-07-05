#!/bin/bash
# mono3d-projection [moderate]: train the fixed mono-3D model with the agent's surface
# (solution/projection.py), then score AP3D / 3D-IoU on the TEST objects in the
# 'moderate' tier. Training is on the full fixed train split; only scoring is sliced.
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/mono3d-detection

python harness.py \
    --task projection \
    --solution solution/projection.py \
    --label moderate \
    --setting moderate \
    --seed ${SEED:-42}
