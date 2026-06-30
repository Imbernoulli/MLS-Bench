#!/bin/bash
set -e
cd /workspace
python "/tests/eval/_inputgen/apply.py" "ml-anomaly-detection" /workspace
ENV=5145e6c31358 SEED=${SEED:-42} OUTPUT_DIR=${OUTPUT_DIR:-./output} \
    python -u scikit-learn/custom_anomaly.py
