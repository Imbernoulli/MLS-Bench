#!/bin/bash
# Scope checkpoints per label: all three test-set labels run concurrently and
# share OUTPUT_DIR, so an unscoped best_model.pt gets clobbered across labels.
export OUTPUT_DIR="${OUTPUT_DIR:-./output}/${ENV:-PDBbind2016}"
# Drop any stale checkpoint from a previous run/iteration.
rm -f "${OUTPUT_DIR}/best_model.pt"
python custom_pla.py \
    --test-set test2016 --data-dir /data \
    --epochs 800 --batch-size 128 --lr 1e-4 --patience 50 \
    --seed ${SEED:-42} --output-dir ${OUTPUT_DIR}
