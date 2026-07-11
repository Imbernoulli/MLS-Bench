#!/bin/bash
# ape-instruction-search (official evaluation): zero-shot SST-2 (2-class sentiment) with
# a FROZEN Qwen2.5-0.5B-Instruct. Same instruction-search surface; different dataset.
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/prompt-optimization-lab

python harness_search.py \
    --solution solution/search.py \
    --dataset sst2 \
    --seed ${SEED:-42}
