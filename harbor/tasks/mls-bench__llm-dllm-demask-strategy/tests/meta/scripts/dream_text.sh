#!/bin/bash
cd /workspace
export TASK_DIR=/workspace/_task
python LLaDA/custom_demask_eval.py \
    --task text \
    --model dream \
    --steps 256 \
    --gen-length 224 \
    --block-length 224 \
    --prefix-len 32 \
    --conf-threshold 0.9 \
    --kl-threshold 0.01 \
    --history-length 2 \
    --temperature 0.0 \
    --n-samples 256 \
    --seed ${SEED:-42} \
    --data-path /workspace/_task/data/c4_texts.json \
    --output-dir ${OUTPUT_DIR:-.}
