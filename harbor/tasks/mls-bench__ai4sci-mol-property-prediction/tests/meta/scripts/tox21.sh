#!/bin/bash
python custom_molprop.py \
    --dataset tox21 --data-dir /data/molecular_property_prediction \
    --epochs 80 --batch-size 32 --lr 1e-4 \
    --seed ${SEED:-42} --output-dir ${OUTPUT_DIR}/${ENV}
