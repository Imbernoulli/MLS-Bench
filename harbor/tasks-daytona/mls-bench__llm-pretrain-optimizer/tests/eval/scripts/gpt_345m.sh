#!/bin/bash
# GPT-2 Medium (24L/16H/1024D, ~355M total params) on ~7.1B tokens (D=20N Chinchilla).
# H100 DDP. custom_pretrain.py divides GRAD_ACCUM by world_size, so the macro
# batch is BATCH_SIZE*GRAD_ACCUM = 576 sequences (589,824 tokens) per optimizer
# step on any world size. The verifier reserves 8 GPUs and passes BATCH_SIZE=36 /
# GRAD_ACCUM=16 (2 micro-steps x 36 per rank); H200 uses 2 GPUs at 96/6. The
# defaults below are the 4-GPU shape (48 x 3 per rank) and keep the same 576.
N_GPU=$(python3 -c "import torch; print(torch.cuda.device_count())")
# Pre-clean stale outputs from a previous test iteration: custom_pretrain.py
# writes these only at the successful end of training and the eval step only
# checks that they exist, so a leftover checkpoint from an earlier run would
# be silently scored if this training run fails.
rm -f "${OUTPUT_DIR:-out}/ckpt_${ENV:-model}.pt" "${OUTPUT_DIR:-out}/model_source_${ENV:-model}.py"
N_LAYER=24 N_HEAD=16 N_EMBD=1024 \
MAX_ITERS=${MAX_ITERS:-12030} EVAL_INTERVAL=${EVAL_INTERVAL:-1000} \
BATCH_SIZE=${BATCH_SIZE:-48} GRAD_ACCUM=${GRAD_ACCUM:-12} LEARNING_RATE=${LEARNING_RATE:-3e-4} \
torchrun --nproc_per_node=${N_GPU} --standalone custom_pretrain.py
