#!/bin/bash
# Train boosted ensemble on Diabetes (regression)
cd /workspace
python scikit-learn/custom_boosting.py \
    --dataset diabetes --task regression \
    --n-rounds 200 --max-depth 3 --learning-rate 0.1 \
    --test-size 0.2 \
    --seed ${SEED:-42} \
    --output-dir ${OUTPUT_DIR:-./output}
