#!/bin/bash
N_GPU=$(python3 -c "import torch; print(torch.cuda.device_count())")
# custom_pretrain.py divides GRAD_ACCUM by WORLD_SIZE under DDP.
# With 4 GPUs this sets env GRAD_ACCUM=24, so actual tokens/iter = 24 * 16 * 1024 = 393,216.
GRAD_ACCUM=$((1536 / (16 * N_GPU)))

N_LAYER=48 N_HEAD=25 N_EMBD=1600 MAX_ITERS=22888 EVAL_INTERVAL=2000 \
BATCH_SIZE=16 GRAD_ACCUM=${GRAD_ACCUM} LEARNING_RATE=2.5e-4 \
torchrun --nproc_per_node=${N_GPU} --standalone custom_pretrain.py
