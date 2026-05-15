#!/bin/bash
# Evaluate bandit policy on linear contextual bandit (K=5, d=10, T=10000)
cd /workspace
python SMPyBandits/custom_bandit.py \
    --env contextual \
    --seed ${SEED:-42} \
    --output-dir ${OUTPUT_DIR:-/tmp/output}
