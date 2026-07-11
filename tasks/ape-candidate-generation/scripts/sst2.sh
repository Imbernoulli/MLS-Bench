#!/bin/bash
# ape-candidate-generation: zero-shot sst2 with a FROZEN Qwen2.5-0.5B-Instruct. Only the
# propose() surface is editable; the harness/executor/splits are fixed.
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/prompt-optimization-lab

python harness_pool.py \
    --solution solution/propose.py --surface propose \
    --dataset sst2 \
    --seed ${SEED:-42}
