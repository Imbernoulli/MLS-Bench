#!/bin/bash
# reid-spatial-pooling: fine-tune a fixed ImageNet-pretrained ResNet-50 on the
# complete official Market-1501 train split for 60 epochs
# with a FIXED loss (triplet + softmax), FIXED P x K sampler and FIXED BNNeck,
# using the agent's spatial pooling module (solution/pooling.py), then score
# single-query mAP / Rank-1 under 3 difficulty settings (higher is better).
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
cd /workspace/torchreid-reid

export REID_DATA="/data/torchreid/market1501_full"
export REID_WEIGHTS="/data/torchreid/weights/resnet50_imagenet.pth"
export REID_EVAL_DATA="${MLSBENCH_TASK_DIR:-/tests/meta}/data/market1501_full"

python harness_pool.py \
    --solution solution/pooling.py \
    --task-id reid-spatial-pooling \
    --seed ${SEED:-42} \
    --epochs 60 \
    --batch-size 64 \
    --num-instances 4
