#!/bin/bash
export WANDB_MODE=disabled

python algorithms/finetune/custom_finetune.py \
    --env pen-cloned-v1 \
    --seed ${SEED:-42} \
    --checkpoints_path "${OUTPUT_DIR:-${SAVE_PATH:-/workspace/saves}/custom}/pen-cloned-v1"
