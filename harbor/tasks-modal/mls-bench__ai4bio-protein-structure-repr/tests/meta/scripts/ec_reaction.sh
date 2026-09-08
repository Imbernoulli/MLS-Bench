#!/bin/bash
# Train and evaluate protein encoder on EC number prediction (384-class multiclass)
cd /workspace

python ProteinWorkshop/custom_protein_encoder.py \
    --task ec_reaction \
    --data-dir /data/ProteinWorkshop \
    --output-dir ${OUTPUT_DIR} \
    --seed ${SEED:-42} \
    --epochs 50 \
    --batch-size 32 \
    --lr 1e-3 \
    --hidden-dim 256 \
    --out-dim 128 \
    --num-layers 6
