#!/bin/bash
export WANDB_MODE=disabled

python algorithms/finetune/custom_finetune.py \
    --env hammer-cloned-v1 \
    --seed ${SEED:-42} \
    --checkpoints_path "${OUTPUT_DIR:-${SAVE_PATH:-/workspace/saves}/custom}/hammer-cloned-v1"
