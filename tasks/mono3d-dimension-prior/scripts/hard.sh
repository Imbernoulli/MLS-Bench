#!/bin/bash
# mono3d-dimension-prior [hard]: train the fixed mono-3D model with the agent's surface
# (solution/dims_prior.py), then score AP3D / 3D-IoU on the TEST objects in the
# 'hard' tier. Training is on the full fixed train split; only scoring is sliced.
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/mono3d-detection

python harness.py \
    --task dims \
    --solution solution/dims_prior.py \
    --label hard \
    --setting hard \
    --seed ${SEED:-42}
