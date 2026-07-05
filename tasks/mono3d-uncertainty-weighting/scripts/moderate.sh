#!/bin/bash
# mono3d-uncertainty-weighting [moderate]: train the fixed mono-3D model with the agent's surface
# (solution/task_weighting.py), then score AP3D / 3D-IoU on the TEST objects in the
# 'moderate' tier. Training is on the full fixed train split; only scoring is sliced.
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/mono3d-detection

python harness.py \
    --task uncertainty \
    --solution solution/task_weighting.py \
    --label moderate \
    --setting moderate \
    --seed ${SEED:-42}
