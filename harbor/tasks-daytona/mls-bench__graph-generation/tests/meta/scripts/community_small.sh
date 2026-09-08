#!/bin/bash
# Train graph generator on community_small dataset (~15-node community graphs)
cd /workspace
python pytorch-geometric/custom_graphgen.py \
    --dataset community_small \
    --epochs 500 \
    --batch-size 32 \
    --seed ${SEED:-42} \
    --output-dir ${OUTPUT_DIR:-./output}
