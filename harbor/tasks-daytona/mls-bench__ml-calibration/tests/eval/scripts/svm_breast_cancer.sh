#!/bin/bash
# Train SVM on Breast Cancer and evaluate calibration
cd /workspace
python scikit-learn/custom_calibration.py \
    --classifier svm --dataset breast_cancer \
    --seed ${SEED:-42} \
    --output-dir ${OUTPUT_DIR:-./output}
