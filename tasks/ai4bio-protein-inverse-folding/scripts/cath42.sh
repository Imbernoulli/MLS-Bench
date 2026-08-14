#!/bin/bash
# Train and evaluate on CATH 4.2 dataset
cd /workspace

# Drop this label's best_model from a previous agent iteration so an
# eval that never improves on it cannot silently score the old model.
rm -f "${OUTPUT_DIR}/best_model_CATH4_2.pt"
PYTHONUNBUFFERED=1 python ProteinInvBench/custom_invfold.py \
    --dataset CATH4.2 --data-root /workspace/data \
    --epochs 100 --batch-size 32 --lr 1e-3 \
    --hidden-dim 128 --num-encoder-layers 3 --k-neighbors 30 \
    --dropout 0.1 --max-train-hours 3.0 \
    --seed ${SEED:-42} --output-dir ${OUTPUT_DIR}
