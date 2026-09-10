#!/bin/bash
# Train GBM on Madelon and evaluate calibration
cd /workspace
python scikit-learn/custom_calibration.py \
    --classifier gbm --dataset madelon \
    --seed ${SEED:-42} \
    --output-dir ${OUTPUT_DIR:-./output}
