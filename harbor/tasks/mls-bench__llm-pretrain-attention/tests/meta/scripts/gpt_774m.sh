#!/bin/bash
N_LAYER=36 N_HEAD=20 N_EMBD=1280 \
MAX_ITERS=23620 EVAL_INTERVAL=2000 \
BATCH_SIZE=16 GRAD_ACCUM=20 LEARNING_RATE=2.5e-4 \
torchrun --nproc_per_node=2 --standalone custom_pretrain.py
