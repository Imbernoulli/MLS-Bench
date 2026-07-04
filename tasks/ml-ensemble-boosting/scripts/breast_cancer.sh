#!/bin/bash
# Train boosted ensemble on the breast_cancer environment.
set -e
cd /workspace

ENV=b6b16eaed5ea SEED=${SEED:-42} OUTPUT_DIR=${OUTPUT_DIR:-./output} \
    python -u scikit-learn/custom_boosting.py
