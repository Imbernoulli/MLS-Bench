#!/bin/bash
set -e
cd /workspace

ENV=206e182f63f7 SEED=${SEED:-42} OUTPUT_DIR=${OUTPUT_DIR:-./output} \
    python -u scikit-learn/custom_anomaly.py
