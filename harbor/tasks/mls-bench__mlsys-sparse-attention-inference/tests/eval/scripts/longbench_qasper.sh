#!/bin/bash
# LongBench Qasper QA eval with Qwen2.5-1.5B-Instruct + agent's SparseAttention.
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
    --env longbench_qasper \
    --model-path /data/qwen2.5-1.5b-instruct \
    --qasper-jsonl /data/longbench-qasper/qasper.jsonl \
    --max-context-len 8192 \
    --max-new-tokens 64 \
    --max-cases ${MAX_CASES:-50} \
    --seed ${SEED:-42} \
    --density-budget 0.25 \
    ${ALLOW_DENSE}
