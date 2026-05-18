#!/bin/bash
# Train on CATH 4.2, evaluate on TS50 (de novo test set)
cd /workspace

PYTHONUNBUFFERED=1 python ProteinInvBench/custom_invfold.py \
    --dataset TS --data-root /workspace/data \
    --epochs 100 --batch-size 32 --lr 1e-3 \
    --hidden-dim 128 --num-encoder-layers 3 --k-neighbors 30 \
    --dropout 0.1 --max-train-hours 6.5 \
    --seed ${SEED:-42} --output-dir ${OUTPUT_DIR}
