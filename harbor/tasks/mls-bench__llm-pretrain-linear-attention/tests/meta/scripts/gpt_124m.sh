#!/bin/bash
# Pre-clean stale outputs from a previous test iteration: custom_pretrain.py
# writes these only at the successful end of training and the eval step only
# checks that they exist, so a leftover checkpoint from an earlier run would
# be silently scored if this training run fails.
rm -f "${OUTPUT_DIR:-out}/ckpt_${ENV:-model}.pt" "${OUTPUT_DIR:-out}/model_source_${ENV:-model}.py"
N_LAYER=12 N_HEAD=12 N_EMBD=768 \
MAX_ITERS=4730 EVAL_INTERVAL=1000 \
BATCH_SIZE=64 GRAD_ACCUM=8 \
python custom_pretrain.py
