#!/bin/bash
set -e
cd /workspace

ENV=satellite SEED=${SEED:-42} OUTPUT_DIR=${OUTPUT_DIR:-./output} \
python "/tests/eval/_inputgen/apply.py" "ml-anomaly-detection" /workspace
    python -u scikit-learn/custom_anomaly.py
