#!/bin/bash
cd /workspace
python scikit-learn/custom_selective.py \
    --dataset law_school \
    --seed ${SEED:-42} \
    --target-coverage 0.8 \
    --output-dir ${OUTPUT_DIR:-./output}
