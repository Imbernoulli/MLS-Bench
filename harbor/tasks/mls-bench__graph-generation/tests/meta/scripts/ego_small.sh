#!/bin/bash
# Train graph generator on ego_small dataset (small ego graphs from Citeseer)
cd /workspace
python pytorch-geometric/custom_graphgen.py \
    --dataset ego_small \
    --epochs 500 \
    --batch-size 32 \
    --seed ${SEED:-42} \
    --output-dir ${OUTPUT_DIR:-./output}
