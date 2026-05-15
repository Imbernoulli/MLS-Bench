#!/bin/bash
# Ill-conditioned linear regression (strongly convex finite-sum problem)

cd /workspace

python opt-vr-bench/custom_vr.py \
    --problem conditioned \
    --seed ${SEED:-42} \
    --output-dir "${OUTPUT_DIR:-./output}"
