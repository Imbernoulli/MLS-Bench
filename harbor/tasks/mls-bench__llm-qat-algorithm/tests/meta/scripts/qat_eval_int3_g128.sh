#!/bin/bash
# INT3 quantization-aware fine-tuning of Pythia-1.4B, group size 128.
# 3-bit (8 levels) is harder than 4-bit and stresses the QAT algorithm more.
set -e

cd /workspace/llm-qat-runtime

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

python custom_qat.py \
    --model-path /data/pythia-1.4b \
    --num-bits 3 \
    --group-size 128 \
    --seed ${SEED:-42} \
    --seqlen 1024
