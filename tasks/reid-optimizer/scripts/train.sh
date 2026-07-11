#!/bin/bash
# reid-optimizer: use the optimizer returned by solution/optimizer.py with the
# harness-owned epoch schedule; sampler, loss, schedule and budget stay fixed.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
cd /workspace/torchreid-reid

export REID_DATA="/data/torchreid/market1501_full"
export REID_WEIGHTS="/data/torchreid/weights/resnet50_imagenet.pth"
export REID_EVAL_DATA="${MLSBENCH_TASK_DIR:-/tests/meta}/data/market1501_full"

python harness_optim.py \
    --solution solution/optimizer.py \
    --task-id reid-optimizer \
    --seed ${SEED:-42} \
    --epochs 60 \
    --batch-size 64 \
    --num-instances 4
