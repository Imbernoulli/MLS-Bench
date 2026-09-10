#!/bin/bash
cd /workspace
export TASK_DIR=/workspace/_task
python LLaDA/custom_demask_eval.py \
    --task math \
    --model llada \
    --steps 256 \
    --gen-length 256 \
    --block-length 64 \
    --conf-threshold 0.6 \
    --kl-threshold 0.01 \
    --history-length 2 \
    --temperature 0.0 \
    --seed ${SEED:-42} \
    --data-path /workspace/_task/data/math_test.json \
    --output-dir ${OUTPUT_DIR:-.}
