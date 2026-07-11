#!/bin/bash
# ape-paraphrase-rewrite: zero-shot sst2 with a FROZEN Qwen2.5-0.5B-Instruct. Only the
# rewrite() surface is editable; the harness/executor/splits are fixed.
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/prompt-optimization-lab

python harness_pool.py \
    --solution solution/rewrite.py --surface rewrite \
    --dataset sst2 \
    --seed ${SEED:-42}
