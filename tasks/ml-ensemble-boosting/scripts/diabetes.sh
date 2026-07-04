#!/bin/bash
# Train boosted ensemble on the diabetes environment.
set -e
cd /workspace

ENV=aca86ca6c724 SEED=${SEED:-42} OUTPUT_DIR=${OUTPUT_DIR:-./output} \
    python -u scikit-learn/custom_boosting.py
