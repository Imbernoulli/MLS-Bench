#!/bin/bash
# reid-batch-mining: fine-tune a FIXED ResNet-50 on a complete official benchmark training split with
# a FIXED batch-hard triplet loss, using the agent's batch sampler
# (solution/sampler.py), then score single-query mAP / Rank-1 (higher is better).
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
cd /workspace/torchreid-reid

export REID_DATA="/data/torchreid/market1501_full"
export REID_WEIGHTS="/data/torchreid/weights/resnet50_imagenet.pth"
export REID_EVAL_DATA="${MLSBENCH_TASK_DIR:-/tests/meta}/data/market1501_full"

python harness_sampler.py \
    --solution solution/sampler.py \
    --task-id reid-batch-mining \
    --seed ${SEED:-42} \
    --epochs 60 \
    --batch-size 64
