#!/bin/bash
# Train perceptual loss flow matching on CIFAR-10 — Large DiT
# hidden_size=768, depth=12, num_heads=12

export TORCH_HOME="${TORCH_HOME:-/data/torch_cache}"
mkdir -p "$TORCH_HOME"

export OUTPUT_DIR="${OUTPUT_DIR:-/result}"
mkdir -p "$OUTPUT_DIR"

export SEED=${SEED:-42}
export NCCL_DEBUG=WARN
export MODEL_HIDDEN_SIZE=768
export MODEL_DEPTH=12
export MODEL_NUM_HEADS=12
export MAX_STEPS=40000
export EVAL_INTERVAL=40000
export BATCH_SIZE=256
export LR=2e-4
export EMA_DECAY=0.999
export NUM_FID_SAMPLES=20000
export NUM_EVAL_STEPS=5

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$NGPU" -gt 1 ]; then
    torchrun --nproc_per_node="$NGPU" --master_port=$((29500 + RANDOM % 1000)) custom_train_perceptual.py
else
    python -u custom_train_perceptual.py
fi
