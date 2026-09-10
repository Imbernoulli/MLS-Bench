#!/bin/bash
export WANDB_MODE=disabled

python algorithms/finetune/custom_finetune.py \
    --env door-cloned-v1 \
    --seed ${SEED:-42} \
    --checkpoints_path "${OUTPUT_DIR:-${SAVE_PATH:-/workspace/saves}/custom}/door-cloned-v1"
