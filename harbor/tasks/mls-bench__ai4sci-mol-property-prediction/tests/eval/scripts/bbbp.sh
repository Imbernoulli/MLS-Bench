#!/bin/bash
# Reference Uni-Mol README: lr=4e-4 bs=128 epoch=40 dropout=0 warmup=0.06.
# We run at bs=32 (single GPU), so linear-scale lr from 4e-4 -> 1e-4.
python custom_molprop.py \
    --dataset bbbp --data-dir /data/molecular_property_prediction \
    --epochs 40 --batch-size 32 --lr 1e-4 \
    --warmup-ratio 0.06 --pooler-dropout 0.0 \
    --seed ${SEED:-42} --output-dir ${OUTPUT_DIR}/${ENV}
