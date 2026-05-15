#!/bin/bash
python custom_molprop.py \
    --dataset bace --data-dir /data/molecular_property_prediction \
    --epochs 60 --batch-size 32 --lr 1e-4 \
    --seed ${SEED:-42} --output-dir ${OUTPUT_DIR}/${ENV}
