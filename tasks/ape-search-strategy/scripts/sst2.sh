#!/bin/bash
# ape-search-strategy (official evaluation): zero-shot SST-2 (2-class sentiment) with a
# FROZEN Qwen2.5-0.5B-Instruct. Same budgeted-search surface; different dataset. The
# agent's select() allocates a fixed dev-execution budget across the FIXED candidate
# pool; the chosen instruction is scored on the held-out TEST set.
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/prompt-optimization-lab

python harness_strategy.py \
    --solution solution/strategy.py \
    --dataset sst2 \
    --budget 200 \
    --seed ${SEED:-42}
