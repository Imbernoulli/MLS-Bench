#!/bin/bash
# Train boosted ensemble on the breast_cancer environment.
set -e
cd /workspace
python "/tests/eval/_inputgen/apply.py" "ml-ensemble-boosting" /workspace
ENV=breast_cancer SEED=${SEED:-42} OUTPUT_DIR=${OUTPUT_DIR:-./output} \
    python -u scikit-learn/custom_boosting.py
