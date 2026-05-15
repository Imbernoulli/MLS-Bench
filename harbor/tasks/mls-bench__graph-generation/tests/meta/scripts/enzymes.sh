#!/bin/bash
# Train graph generator on ENZYMES dataset (protein structure graphs)
cd /workspace
python pytorch-geometric/custom_graphgen.py \
    --dataset enzymes \
    --epochs 500 \
    --batch-size 32 \
    --seed ${SEED:-42} \
    --output-dir ${OUTPUT_DIR:-./output}
