#!/bin/bash
# INT4 quantization-aware fine-tuning of Pythia-1.4B, group size 128.
# Loads pretrained Pythia-1.4B weights, runs QAT finetune on WikiText-2,
# then applies a real INT4 quantize-dequantize roundtrip and evaluates ppl.
set -e

cd /workspace/llm-qat-runtime

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

python custom_qat.py \
    --model-path /data/pythia-1.4b \
    --num-bits 4 \
    --group-size 128 \
    --seed ${SEED:-42} \
    --seqlen 1024
