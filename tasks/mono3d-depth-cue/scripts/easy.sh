#!/bin/bash
# mono3d-depth-cue [easy]: train the fixed mono-3D model with the agent's surface
# (solution/depth_cue.py), then score AP3D / 3D-IoU on the TEST objects in the
# 'easy' tier. Training is on the full fixed train split; only scoring is sliced.
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/mono3d-detection

python harness.py \
    --task depth_cue \
    --solution solution/depth_cue.py \
    --label easy \
    --setting easy \
    --seed ${SEED:-42}
