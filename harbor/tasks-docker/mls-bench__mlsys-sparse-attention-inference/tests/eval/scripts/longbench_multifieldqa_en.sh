#!/bin/bash
# LongBench MultiFieldQA-EN eval with Qwen2.5-1.5B-Instruct + agent's SparseAttention.
set -e
cd /workspace

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

ALLOW_DENSE=""
if [ "${ALLOW_DENSE_FLAG:-0}" = "1" ]; then
    ALLOW_DENSE="--allow-dense"
fi

python -u sparse-attn-eval/run_llm.py \
    --env longbench_multifieldqa_en \
    --model-path /data/qwen2.5-1.5b-instruct \
    --multifieldqa-jsonl /data/longbench-qasper/multifieldqa_en.jsonl \
    --max-context-len 8192 \
    --max-new-tokens 64 \
    --max-cases ${MAX_CASES:-100} \
    --seed ${SEED:-42} \
    --density-budget 0.25 \
    ${ALLOW_DENSE}
