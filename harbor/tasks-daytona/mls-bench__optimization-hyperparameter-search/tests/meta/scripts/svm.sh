#!/bin/bash
# Run HPO strategy on SVM tuning benchmark (Breast Cancer)

cd /workspace

python scikit-learn/custom_hpo.py \
    --benchmark svm \
    --seed ${SEED:-42} \
    --output-dir ${OUTPUT_DIR:-./output}
