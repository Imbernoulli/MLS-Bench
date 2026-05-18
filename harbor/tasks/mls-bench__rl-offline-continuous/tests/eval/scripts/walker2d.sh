#!/bin/bash
export WANDB_MODE=disabled

python algorithms/offline/custom.py \
    --env walker2d-medium-v2 \
    --seed ${SEED:-42} \
    --checkpoints_path "${OUTPUT_DIR:-${SAVE_PATH:-/workspace/saves}/custom}/walker2d-medium-v2"