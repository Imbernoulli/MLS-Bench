#!/bin/bash
set -e
cd /workspace

ENV=shuttle SEED=${SEED:-42} OUTPUT_DIR=${OUTPUT_DIR:-./output} \
    python -u scikit-learn/custom_anomaly.py
