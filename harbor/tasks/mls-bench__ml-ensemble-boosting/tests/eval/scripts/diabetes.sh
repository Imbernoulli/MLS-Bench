#!/bin/bash
# Train boosted ensemble on the diabetes environment.
set -e
cd /workspace
python "/tests/eval/_inputgen/apply.py" "ml-ensemble-boosting" /workspace
ENV=diabetes SEED=${SEED:-42} OUTPUT_DIR=${OUTPUT_DIR:-./output} \
    python -u scikit-learn/custom_boosting.py
