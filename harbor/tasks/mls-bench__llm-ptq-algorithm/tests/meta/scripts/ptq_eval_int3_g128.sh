#!/bin/bash
# INT3 quantization with group size 128 -- harder PTQ setting.
# 3-bit quantization is much more challenging (only 8 discrete levels vs 16 for INT4),
# providing better differentiation between algorithms.
set -e

# workdir is already /workspace/gptq

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

python custom_ptq.py \
    --model-path /data/mistral-7b-v01 \
    --num-bits 3 \
    --group-size 128 \
    --nsamples 128 \
    --seed ${SEED:-42} \
    --seqlen 2048
