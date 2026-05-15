#!/bin/bash
# INT4 quantization with group size 64 -- finer granularity PTQ setting.
# Smaller group size means more quantization parameters but tighter constraints
# per group, testing whether algorithms generalize across granularity levels.
set -e

# workdir is already /workspace/gptq

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

python custom_ptq.py \
    --model-path /data/mistral-7b-v01 \
    --num-bits 4 \
    --group-size 64 \
    --nsamples 128 \
    --seed ${SEED:-42} \
    --seqlen 2048
