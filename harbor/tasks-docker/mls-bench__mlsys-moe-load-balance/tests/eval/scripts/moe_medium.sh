#!/bin/bash
# Evaluate EPLB on Qwen3-MoE config (128 experts, 32 GPUs, 4 nodes)

cd /workspace

python eplb/custom_eplb.py \
    --config qwen3-moe \
    --seed ${SEED:-42} \
    --output-dir ${OUTPUT_DIR:-.} \
    --num-trials 10 \
    --num-timing 20
