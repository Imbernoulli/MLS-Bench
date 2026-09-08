#!/bin/bash
export WANDB_MODE=disabled

python algorithms/offline/custom.py \
    --env maze2d-medium-v1 \
    --seed ${SEED:-42} \
    --checkpoints_path "${OUTPUT_DIR:-${SAVE_PATH:-/workspace/saves}/custom}/maze2d-medium-v1"