#!/bin/bash
# reid-reranking: train a FIXED ResNet-50 on a complete official benchmark training split with a FIXED
# loss / sampler / budget, then apply the agent's TEST-TIME re-ranking
# (solution/rerank.py) to the query-gallery distances and score single-query
# mAP / Rank-1 under 3 difficulty settings (higher is better).
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
cd /workspace/torchreid-reid

export REID_DATA="/data/torchreid/market1501_full"
export REID_WEIGHTS="/data/torchreid/weights/resnet50_imagenet.pth"
export REID_EVAL_DATA="${MLSBENCH_TASK_DIR:-/tests/meta}/data/market1501_full"

python harness_rerank.py \
    --solution solution/rerank.py \
    --task-id reid-reranking \
    --seed ${SEED:-42} \
    --epochs 60 \
    --batch-size 64 \
    --num-instances 4
