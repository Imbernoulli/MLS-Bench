#!/bin/bash
# ape-search-algorithm: zero-shot agnews with a FROZEN Qwen2.5-0.5B-Instruct. Only the
# search() surface is editable; the harness/executor/splits are fixed.
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/prompt-optimization-lab

python harness_searchalgo.py \
    --solution solution/searchalgo.py --mode search \
    --dataset agnews --budget 120 \
    --seed ${SEED:-42}
