#!/bin/bash
# Train and evaluate protein encoder on Fold classification (1195-class multiclass)
cd /workspace

python ProteinWorkshop/custom_protein_encoder.py \
    --task fold_fold \
    --data-dir /data/ProteinWorkshop \
    --output-dir ${OUTPUT_DIR} \
    --seed ${SEED:-42} \
    --epochs 150 \
    --batch-size 32 \
    --lr 1e-3 \
    --hidden-dim 256 \
    --out-dim 128 \
    --num-layers 6
