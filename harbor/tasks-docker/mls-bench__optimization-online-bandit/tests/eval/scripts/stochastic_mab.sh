#!/bin/bash
# Evaluate bandit policy on stochastic 10-armed Bernoulli bandit (T=10000)
cd /workspace
python SMPyBandits/custom_bandit.py \
    --env stochastic_mab \
    --seed ${SEED:-42} \
    --output-dir ${OUTPUT_DIR:-/tmp/output}
