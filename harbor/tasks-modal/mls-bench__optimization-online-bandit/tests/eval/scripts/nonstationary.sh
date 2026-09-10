#!/bin/bash
# Evaluate bandit policy on non-stationary piece-wise Bernoulli bandit (T=10000)
cd /workspace
python SMPyBandits/custom_bandit.py \
    --env nonstationary \
    --seed ${SEED:-42} \
    --output-dir ${OUTPUT_DIR:-/tmp/output}
