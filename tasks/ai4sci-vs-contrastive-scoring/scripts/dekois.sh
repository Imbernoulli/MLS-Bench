#!/bin/bash
# Evaluate on DEKOIS 2.0 benchmark.


CHECKPOINT="${OUTPUT_DIR}/checkpoints_no_similar_protein/checkpoint_best.pt"

# Fail loudly if training did not produce a checkpoint. scripts/train.sh wipes
# the save dir before every run, so a missing checkpoint_best.pt means this
# test's training step crashed or was killed. Without this guard the evaluation
# could only ever score a stale checkpoint or die with an opaque torch error.
if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: checkpoint not found: ${CHECKPOINT}" >&2
    echo "ERROR: training (scripts/train.sh) must complete successfully before evaluation; aborting DEKOIS eval." >&2
    exit 1
fi

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
    --test-task DEKOIS \
    2>&1
