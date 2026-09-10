#!/bin/bash
# Ill-conditioned linear regression (strongly convex finite-sum problem)
cd /workspace
python "/tests/eval/_inputgen/apply.py" "optimization-variance-reduction" /workspace
python opt-vr-bench/vr_driver.py \
    --problem conditioned \
    --seed ${SEED:-42} \
    --output-dir "${OUTPUT_DIR:-./output}"
