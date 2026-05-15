#!/bin/bash
# Run HPO strategy on XGBoost tuning benchmark (California Housing)

cd /workspace

python scikit-learn/custom_hpo.py \
    --benchmark xgboost \
    --seed ${SEED:-42} \
    --output-dir ${OUTPUT_DIR:-./output}
