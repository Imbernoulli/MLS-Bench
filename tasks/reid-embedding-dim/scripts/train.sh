#!/bin/bash
# reid-embedding-dim: use a fixed Linear+BN embedding head whose output dimension
# is selected by solution/dimension.py; all other training components are fixed.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
cd /workspace/torchreid-reid

export REID_DATA="/data/torchreid/market1501_full"
export REID_WEIGHTS="/data/torchreid/weights/resnet50_imagenet.pth"
export REID_EVAL_DATA="${MLSBENCH_TASK_DIR:-/tests/meta}/data/market1501_full"

python harness_dim.py \
    --solution solution/dimension.py \
    --task-id reid-embedding-dim \
    --seed ${SEED:-42} \
    --epochs 60 \
    --batch-size 64 \
    --num-instances 4
