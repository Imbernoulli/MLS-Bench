#!/bin/bash
# Train unconditional DDPM on CIFAR-10 — Large (~140M params)
# UNet2DModel: block_out_channels=(256,512,512,512), layers_per_block=2
# 8-GPU DDP training

export TORCH_HOME="${TORCH_HOME:-/data/pretrained}"
mkdir -p "$TORCH_HOME"

export OUTPUT_DIR="${OUTPUT_DIR:-/result}"
mkdir -p "$OUTPUT_DIR"

export SEED=${SEED:-42}
export BLOCK_OUT_CHANNELS="160,320,320,320"
export LAYERS_PER_BLOCK=2
export MAX_STEPS=35000
export EVAL_INTERVAL=10000
export EMA_RATE=0.9995
export BATCH_SIZE=128
export LR=2e-4
export NUM_FID_SAMPLES=50000
export DIFFUSION_STEPS=1000
export SAMPLE_STEPS=50

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$NGPU" -gt 1 ]; then
    torchrun --nproc_per_node="$NGPU" --master_port=29500 custom_train.py
else
    python -u custom_train.py
fi
