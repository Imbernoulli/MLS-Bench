#!/bin/bash
# Logistic regression on MNIST (convex finite-sum problem)
cd /workspace
python "/tests/eval/_inputgen/apply.py" "optimization-variance-reduction" /workspace
python opt-vr-bench/vr_driver.py \
    --problem logistic \
    --seed ${SEED:-42} \
    --output-dir "${OUTPUT_DIR:-./output}"
