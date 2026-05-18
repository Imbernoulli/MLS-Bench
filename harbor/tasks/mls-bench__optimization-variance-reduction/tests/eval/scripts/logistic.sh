#!/bin/bash
# Logistic regression on MNIST (convex finite-sum problem)

cd /workspace

python opt-vr-bench/custom_vr.py \
    --problem logistic \
    --seed ${SEED:-42} \
    --output-dir "${OUTPUT_DIR:-./output}"
