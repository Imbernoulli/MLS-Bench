#!/bin/bash
# Train VAE on CIFAR-10 — Large (~55M params)
# AutoencoderKL: block_out_channels=(128,256,512), layers_per_block=2, latent 8x8x4


# Verifier-only: build _verifier_custom_train.py — the agent's file plus the
# race fix when the pre-fix block is still present. The agent's file
# itself is never modified, keeping the pristine guard valid on
# verifier re-runs (see _runtime_patch.sh).
source "$(dirname "${BASH_SOURCE[0]}")/_runtime_patch.sh"

export OUTPUT_DIR="${OUTPUT_DIR:-/result}"
mkdir -p "$OUTPUT_DIR"

export SEED=${SEED:-42}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="${TORCH_HOME:-/data/pretrained}"
mkdir -p "$TORCH_HOME"
export BLOCK_OUT_CHANNELS="128,256,512"
export LATENT_CHANNELS=16
export LAYERS_PER_BLOCK=2
export NCCL_DEBUG=WARN
export MAX_STEPS=30000
export EVAL_INTERVAL=5000
export EMA_RATE=0.999
export BATCH_SIZE=128
export LR=2e-4

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$NGPU" -gt 1 ]; then
    torchrun --nproc_per_node="$NGPU" --master_port=$((29500 + RANDOM % 1000)) _verifier_custom_train.py
else
    python -u _verifier_custom_train.py
fi
