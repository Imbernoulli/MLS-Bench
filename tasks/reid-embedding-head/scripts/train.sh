#!/bin/bash
# reid-embedding-head: fine-tune a FIXED ResNet-50 on a complete official benchmark training split
# with a FIXED loss (triplet + softmax) and FIXED P x K sampler, using the agent's
# embedding head (solution/head.py), then score single-query mAP / Rank-1.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
cd /workspace/torchreid-reid

export REID_DATA="/data/torchreid/market1501_full"
export REID_WEIGHTS="/data/torchreid/weights/resnet50_imagenet.pth"
export REID_EVAL_DATA="${MLSBENCH_TASK_DIR:-/tests/meta}/data/market1501_full"

python harness_head.py \
    --solution solution/head.py \
    --task-id reid-embedding-head \
    --seed ${SEED:-42} \
    --epochs 60 \
    --batch-size 64 \
    --num-instances 4
