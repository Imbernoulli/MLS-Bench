#!/bin/bash
# mono3d-depth-cue [hard]: train the fixed mono-3D model with the agent's surface
# (solution/depth_cue.py), then score AP3D / 3D-IoU on the TEST objects in the
# 'hard' tier. Training is on the full fixed train split; only scoring is sliced.
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/mono3d-detection

python harness.py \
    --task depth_cue \
    --solution solution/depth_cue.py \
    --label hard \
    --setting hard \
    --seed ${SEED:-42}
