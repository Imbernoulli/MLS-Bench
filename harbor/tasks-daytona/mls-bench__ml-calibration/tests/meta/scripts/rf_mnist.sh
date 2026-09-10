#!/bin/bash
# Train Random Forest on MNIST and evaluate calibration
cd /workspace
python scikit-learn/custom_calibration.py \
    --classifier rf --dataset mnist \
    --seed ${SEED:-42} \
    --output-dir ${OUTPUT_DIR:-./output}
