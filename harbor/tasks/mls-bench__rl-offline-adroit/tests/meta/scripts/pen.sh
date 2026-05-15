#!/bin/bash
export WANDB_MODE=disabled

python algorithms/offline/custom_adroit.py \
    --env pen-human-v1 \
    --seed ${SEED:-42} \
    --checkpoints_path "${OUTPUT_DIR:-${SAVE_PATH:-/workspace/saves}/custom}/pen-human-v1"
