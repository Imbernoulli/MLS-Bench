#!/bin/bash
# Evaluate EPLB on DeepSeek-V3 config (256 experts, 64 GPUs, 8 nodes)

cd /workspace

python eplb/custom_eplb.py \
    --config deepseek-v3 \
    --seed ${SEED:-42} \
    --output-dir ${OUTPUT_DIR:-.} \
    --num-trials 10 \
    --num-timing 20
