#!/bin/bash
# Train boosted ensemble on the california_housing environment.
set -e
cd /workspace
python "/tests/eval/_inputgen/apply.py" "ml-ensemble-boosting" /workspace
ENV=california_housing SEED=${SEED:-42} OUTPUT_DIR=${OUTPUT_DIR:-./output} \
    python -u scikit-learn/custom_boosting.py
