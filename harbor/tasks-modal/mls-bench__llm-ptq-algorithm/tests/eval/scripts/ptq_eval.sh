#!/bin/bash
# Post-training quantization evaluation for Mistral-7B-v0.1.
# Loads real Mistral-7B weights from /data/mistral-7b-v01 (pre-downloaded),
# applies the PTQ algorithm from custom_ptq.py, and evaluates perplexity
# on WikiText-2.
# Single GPU, no training needed.
set -e

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
    --nsamples 128 \
    --seed ${SEED:-42} \
    --seqlen 2048
