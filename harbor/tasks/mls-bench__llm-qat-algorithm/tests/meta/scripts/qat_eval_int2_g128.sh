#!/bin/bash
# INT2 quantization-aware fine-tuning of Pythia-1.4B, group size 128.
# 2-bit (4 levels) is an extreme low-bit setting where naive QAT collapses.
set -e

cd /workspace/llm-qat-runtime

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

python custom_qat.py \
    --model-path /data/pythia-1.4b \
    --num-bits 2 \
    --group-size 128 \
    --seed ${SEED:-42} \
    --seqlen 1024
