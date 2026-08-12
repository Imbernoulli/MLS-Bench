#!/bin/bash
# Train class-conditional DDPM on CIFAR-10 — Medium (~36M params)
# UNet2DModel: block_out_channels=(128,256,256,256), layers_per_block=2
# Same architecture as google/ddpm-cifar10-32
# 8-GPU DDP training

export TORCH_HOME="${TORCH_HOME:-/data/pretrained}"
mkdir -p "$TORCH_HOME"

export OUTPUT_DIR="${OUTPUT_DIR:-/result}"
mkdir -p "$OUTPUT_DIR"

export SEED=${SEED:-42}
export MAX_STEPS=35000
export EVAL_INTERVAL=35000
export EMA_RATE=0.9995
export BATCH_SIZE=128
export LR=2e-4
export NUM_FID_SAMPLES=50000
export NUM_CLASSES=10
export DIFFUSION_STEPS=1000
export SAMPLE_STEPS=50

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]; then
    IFS=',' read -ra _MLSBENCH_GPUS <<< "$CUDA_VISIBLE_DEVICES"
    NGPU=0
    for _gpu in "${_MLSBENCH_GPUS[@]}"; do
        if [ -n "$_gpu" ]; then
            NGPU=$((NGPU + 1))
        fi
    done
else
    NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
if [ "$NGPU" -gt 1 ]; then
    torchrun --nproc_per_node="$NGPU" --master_port=$((29500 + ($(printf "%s" "cv-diffusion-conditioning-${ENV:-x}-${SEED:-42}-${OUTPUT_DIR:-}" | cksum | cut -d' ' -f1) % 1000))) custom_train.py
else
    python -u custom_train.py
fi
