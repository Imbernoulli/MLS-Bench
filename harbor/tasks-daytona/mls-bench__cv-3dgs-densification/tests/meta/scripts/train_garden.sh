#!/bin/bash
# Train 3DGS on Mip-NeRF 360 garden scene (outdoor, 8x downsampled)

export SEED=${SEED:-42}
export OUTPUT_DIR="${OUTPUT_DIR:-/result}"
mkdir -p "$OUTPUT_DIR"

# Redirect temp files to workspace (host fs) to avoid filling tmpfs
export TMPDIR=/tmp/gsplat_tmp
# export HOME kept as default
export TORCH_HOME="${TORCH_HOME:-/data/torch_cache}"
mkdir -p "$TMPDIR"
mkdir -p "$TORCH_HOME"

# cd handled by submit.sh

python train_gsplat.py \
    --data_dir "${DATA_DIR:-/data/360_v2}/garden" \
    --data_factor 8 \
    --result_dir $OUTPUT_DIR \
    --max_steps 30000 \
    --eval_steps 7000 30000 \
    --seed "$SEED"
