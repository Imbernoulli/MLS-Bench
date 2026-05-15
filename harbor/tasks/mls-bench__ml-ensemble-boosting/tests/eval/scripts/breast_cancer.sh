#!/bin/bash
# Train boosted ensemble on Breast Cancer (classification)
cd /workspace
python scikit-learn/custom_boosting.py \
    --dataset breast_cancer --task classification \
    --n-rounds 200 --max-depth 3 --learning-rate 0.1 \
    --test-size 0.2 \
    --seed ${SEED:-42} \
    --output-dir ${OUTPUT_DIR:-./output}
