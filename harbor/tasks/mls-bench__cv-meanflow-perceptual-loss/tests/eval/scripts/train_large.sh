#!/bin/bash
# Train perceptual loss flow matching on CIFAR-10 — Large DiT (~260M params)
# hidden_size=1024, depth=12, num_heads=16

export TORCH_HOME="${TORCH_HOME:-/data/torch_cache}"
mkdir -p "$TORCH_HOME"

export OUTPUT_DIR="${OUTPUT_DIR:-/result}"
mkdir -p "$OUTPUT_DIR"

export SEED=${SEED:-42}
export MODEL_HIDDEN_SIZE=1024
export MODEL_DEPTH=12
export MODEL_NUM_HEADS=16
export MAX_STEPS=40000
export EVAL_INTERVAL=40000
export BATCH_SIZE=32
export LR=5e-5
export NUM_FID_SAMPLES=2048
export NUM_EVAL_STEPS=10

python -u custom_train_perceptual.py
