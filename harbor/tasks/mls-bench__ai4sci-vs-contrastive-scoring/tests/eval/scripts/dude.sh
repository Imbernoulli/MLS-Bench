#!/bin/bash
# Evaluate on DUD-E benchmark (102 protein targets).


CHECKPOINT="${OUTPUT_DIR}/checkpoints_no_similar_protein/checkpoint_best.pt"
RESULTS="${OUTPUT_DIR}/results"
mkdir -p "${RESULTS}"

LOCAL_UNIMOL="./unimol"
export PYTHONPATH="${LOCAL_UNIMOL}:$PYTHONPATH"

DATA_ROOT="/data/test_datasets"

CUDA_VISIBLE_DEVICES=0 python "${LOCAL_UNIMOL}/test.py" \
    "${DATA_ROOT}" \
    --user-dir "${LOCAL_UNIMOL}" \
    --valid-subset test \
    --results-path "${RESULTS}" \
    --num-workers 0 \
    --ddp-backend c10d \
    --distributed-world-size 1 \
    --batch-size 256 \
    --task test_task \
    --loss custom_vs_loss \
    --arch custom_vs_model \
    --fp16 \
    --fp16-init-scale 4 \
    --fp16-scale-window 256 \
    --seed ${SEED:-1} \
    --path "${CHECKPOINT}" \
    --log-interval 100 \
    --log-format simple \
    --max-pocket-atoms 511 \
    --test-task DUDE \
    2>&1
