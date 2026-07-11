#!/bin/bash
# ape-search-strategy: zero-shot AG News (4-class) with a FROZEN Qwen2.5-0.5B-
# Instruct. The candidate POOL and the estimator (dev execution-accuracy) are FIXED;
# the agent's select() is the budgeted SEARCH/ALLOCATION over a strict cap of
# (candidate, dev-example) executions. The chosen instruction is scored on the
# HELD-OUT TEST set. Everything but the search allocation is fixed.
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/prompt-optimization-lab

python harness_strategy.py \
    --solution solution/strategy.py \
    --dataset agnews \
    --budget 200 \
    --seed ${SEED:-42}
