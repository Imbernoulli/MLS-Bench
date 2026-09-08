#!/bin/bash
# Evaluate EPLB on DeepSeek-V2 config (160 experts, 32 GPUs, 4 nodes)

cd /workspace

python eplb/custom_eplb.py \
    --config deepseek-v2 \
    --seed ${SEED:-42} \
    --output-dir ${OUTPUT_DIR:-.} \
    --num-trials 10 \
    --num-timing 20
