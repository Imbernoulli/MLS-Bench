#!/bin/bash
# Train perceptual loss flow matching on CIFAR-10 — Small DiT (~40M params)
# hidden_size=512, depth=8, num_heads=8

export TORCH_HOME="${TORCH_HOME:-/data/torch_cache}"
mkdir -p "$TORCH_HOME"

export OUTPUT_DIR="${OUTPUT_DIR:-/result}"
mkdir -p "$OUTPUT_DIR"

export SEED=${SEED:-42}
export MODEL_HIDDEN_SIZE=512
export MODEL_DEPTH=8
export MODEL_NUM_HEADS=8
export MAX_STEPS=10000
export EVAL_INTERVAL=10000
export BATCH_SIZE=128
export LR=1e-4
export NUM_FID_SAMPLES=2048
export NUM_EVAL_STEPS=10

python -u custom_train_perceptual.py
