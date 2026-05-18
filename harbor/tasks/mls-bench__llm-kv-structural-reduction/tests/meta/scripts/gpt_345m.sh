#!/bin/bash
# GPT-2 Medium (24L/16H/1024D, ~345M params) on ~7.1B tokens (Chinchilla-optimal).
# 2-GPU DDP, matching llm-pretrain-attention setup.
cd "${MLSBENCH_PKG_DIR:-/workspace/nanoGPT}"
N_GPU=$(python3 -c "import torch; print(torch.cuda.device_count())")
SEED="${SEED:-42}" OUTPUT_DIR="${OUTPUT_DIR:-out}" ENV="${ENV:-gpt-345m}" \
RUN_AUX_EVAL=1 AUX_EVAL_DATASETS="${AUX_EVAL_DATASETS:-wikitext2,wikitext103,lambada}" \
AUX_EVAL_ITERS="${AUX_EVAL_ITERS:-16}" AUX_EVAL_BATCH_SIZE="${AUX_EVAL_BATCH_SIZE:-4}" \
N_LAYER=24 N_HEAD=16 N_EMBD=1024 \
MAX_ITERS="${MAX_ITERS:-13535}" EVAL_INTERVAL="${EVAL_INTERVAL:-1000}" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
BATCH_SIZE="${BATCH_SIZE:-32}" GRAD_ACCUM="${GRAD_ACCUM:-16}" LEARNING_RATE="${LEARNING_RATE:-3e-4}" \
torchrun --nproc_per_node="${N_GPU}" --standalone custom_pretrain.py
