#!/bin/bash
# Train perceptual loss flow matching on CIFAR-10 — Medium DiT (~100M params)
# hidden_size=768, depth=12, num_heads=12

export TORCH_HOME="${TORCH_HOME:-/data/torch_cache}"
mkdir -p "$TORCH_HOME"

export OUTPUT_DIR="${OUTPUT_DIR:-/result}"
mkdir -p "$OUTPUT_DIR"

export SEED=${SEED:-42}
export MODEL_HIDDEN_SIZE=768
export MODEL_DEPTH=12
export MODEL_NUM_HEADS=12
export MAX_STEPS=20000
export EVAL_INTERVAL=20000
export BATCH_SIZE=64
export LR=1e-4
export NUM_FID_SAMPLES=2048
export NUM_EVAL_STEPS=10

python -u custom_train_perceptual.py
