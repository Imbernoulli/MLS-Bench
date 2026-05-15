#!/bin/bash
cd /workspace
export TASK_DIR=/workspace/_task
python LLaDA/custom_demask_eval.py \
    --task humaneval \
    --model llada \
    --steps 256 \
    --gen-length 256 \
    --block-length 64 \
    --conf-threshold 0.9 \
    --kl-threshold 0.01 \
    --history-length 2 \
    --temperature 0.0 \
    --seed ${SEED:-42} \
    --data-path /workspace/_task/data/HumanEval.jsonl.gz \
    --output-dir ${OUTPUT_DIR:-.}
