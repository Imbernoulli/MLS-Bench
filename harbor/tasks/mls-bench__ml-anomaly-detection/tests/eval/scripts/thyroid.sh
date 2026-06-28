#!/bin/bash
set -e
cd /workspace
python "/tests/eval/_inputgen/apply.py" "ml-anomaly-detection" /workspace
ENV=thyroid SEED=${SEED:-42} OUTPUT_DIR=${OUTPUT_DIR:-./output} \
    python -u scikit-learn/custom_anomaly.py
