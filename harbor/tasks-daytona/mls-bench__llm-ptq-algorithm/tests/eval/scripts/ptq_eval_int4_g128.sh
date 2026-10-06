#!/bin/bash
# INT4 quantization with group size 128 -- standard PTQ setting.
# Loads Mistral-7B-v0.1 weights, applies the PTQ algorithm, evaluates on WikiText-2.
set -e

# workdir is already /workspace/gptq

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
source "$(dirname "${BASH_SOURCE[0]}")/_withheld.sh"
HF_DATASETS_CACHE="$(mlsb_eval_wikitext "${HF_DATASETS_CACHE:-/data/wikitext2}")" || exit 1
export HF_DATASETS_CACHE
# The scored perplexity is the WikiText-2 TEST split; never fall back to validation here.
export MLSB_REQUIRE_TEST_SPLIT=1

python custom_ptq.py \
    --model-path /data/mistral-7b-v01 \
    --num-bits 4 \
    --group-size 128 \
    --nsamples 128 \
    --seed ${SEED:-42} \
    --seqlen 2048
