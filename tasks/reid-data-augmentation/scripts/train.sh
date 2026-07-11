#!/bin/bash
# reid-data-augmentation: fine-tune a FIXED ResNet-50 on a complete official benchmark training split
# with a FIXED loss (triplet + softmax) and FIXED P x K sampler, applying the
# agent's TRAIN-TIME augmentation (solution/augment.py), then score single-query
# mAP / Rank-1 under 3 difficulty settings (higher is better).
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
cd /workspace/torchreid-reid

export REID_DATA="/data/torchreid/market1501_full"
export REID_WEIGHTS="/data/torchreid/weights/resnet50_imagenet.pth"
export REID_EVAL_DATA="${MLSBENCH_TASK_DIR:-/tests/meta}/data/market1501_full"

python harness_aug.py \
    --solution solution/augment.py \
    --task-id reid-data-augmentation \
    --seed ${SEED:-42} \
    --epochs 60 \
    --batch-size 64 \
    --num-instances 4
