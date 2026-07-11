#!/bin/bash
# ape-instruction-search: zero-shot AG News (4-class) with a FROZEN Qwen2.5-0.5B-
# Instruct. The agent's optimize() proposes+selects an INSTRUCTION on the dev set;
# accuracy is measured on the HELD-OUT TEST set by the LM's forced-choice per-label
# likelihood. Demonstrations are absent from the executed prompt; everything but the
# instruction is fixed.
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/prompt-optimization-lab

python harness_search.py \
    --solution solution/search.py \
    --dataset agnews \
    --seed ${SEED:-42}
