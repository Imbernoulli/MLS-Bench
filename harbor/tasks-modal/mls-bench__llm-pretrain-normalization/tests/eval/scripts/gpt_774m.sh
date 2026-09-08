#!/bin/bash
# Pre-clean stale outputs from a previous test iteration: custom_pretrain.py
# writes these only at the successful end of training and the eval step only
# checks that they exist, so a leftover checkpoint from an earlier run would
# be silently scored if this training run fails.
rm -f "${OUTPUT_DIR:-out}/ckpt_${ENV:-model}.pt" "${OUTPUT_DIR:-out}/model_source_${ENV:-model}.py"
N_LAYER=36 N_HEAD=20 N_EMBD=1280 \
MAX_ITERS=23620 EVAL_INTERVAL=2000 \
BATCH_SIZE=16 GRAD_ACCUM=20 LEARNING_RATE=2.5e-4 \
torchrun --nproc_per_node=2 --standalone custom_pretrain.py
