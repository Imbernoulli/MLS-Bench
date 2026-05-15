#!/bin/bash
N_GPU=$(python3 -c "import torch; print(torch.cuda.device_count())")
BATCH_SIZE=${BATCH_SIZE:-32}
GRAD_ACCUM=${GRAD_ACCUM:-16}
MAX_ITERS=${MAX_ITERS:-29068}
EVAL_INTERVAL=${EVAL_INTERVAL:-4000}
LEARNING_RATE=${LEARNING_RATE:-2.5e-4}

# GPT-2 Large style shape (~774M total params, ~762M non-embedding) on ~15.2B tokens (D=20N).
# custom_pretrain.py divides GRAD_ACCUM by WORLD_SIZE under DDP, so keep GRAD_ACCUM divisible by N_GPU.
N_LAYER=36 N_HEAD=20 N_EMBD=1280 MAX_ITERS=${MAX_ITERS} EVAL_INTERVAL=${EVAL_INTERVAL} \
BATCH_SIZE=${BATCH_SIZE} GRAD_ACCUM=${GRAD_ACCUM} LEARNING_RATE=${LEARNING_RATE} \
torchrun --nproc_per_node=${N_GPU} --standalone custom_pretrain.py
