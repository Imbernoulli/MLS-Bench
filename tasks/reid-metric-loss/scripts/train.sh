#!/bin/bash
# reid-metric-loss: fine-tune a FIXED ResNet-50 on a complete official benchmark training split with
# the agent's training loss (solution/loss.py) for a fixed budget, then score
# single-query mAP / Rank-1 (higher is better).
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
cd /workspace/torchreid-reid

export REID_DATA="/data/torchreid/market1501_full"
export REID_WEIGHTS="/data/torchreid/weights/resnet50_imagenet.pth"
export REID_EVAL_DATA="${MLSBENCH_TASK_DIR:-/tests/meta}/data/market1501_full"

python harness_loss.py \
    --solution solution/loss.py \
    --task-id reid-metric-loss \
    --seed ${SEED:-42} \
    --epochs 60 \
    --batch-size 64 \
    --num-instances 4
