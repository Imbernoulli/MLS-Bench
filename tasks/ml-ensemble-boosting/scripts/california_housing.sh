#!/bin/bash
# Train boosted ensemble on the california_housing environment.
set -e
cd /workspace

ENV=16d2b29d7afb SEED=${SEED:-42} OUTPUT_DIR=${OUTPUT_DIR:-./output} \
    python -u scikit-learn/custom_boosting.py
