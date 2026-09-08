#!/bin/bash
cd /workspace/scaling-law-lab
python custom_scaling_law.py \
    --benchmark sld-lrbsz \
    --seed ${SEED:-42} \
    --output-dir ${OUTPUT_DIR:-./output}
