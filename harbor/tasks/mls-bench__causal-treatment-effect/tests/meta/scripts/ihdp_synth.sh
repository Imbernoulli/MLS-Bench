#!/bin/bash
# Evaluate CATE estimator on an explicitly synthetic IHDP-inspired DGP.
cd /workspace
python "/tests/eval/_inputgen/apply.py" "causal-treatment-effect" /workspace
python scikit-learn/custom_cate.py \
    --dataset ihdp_synth \
    --seed ${SEED:-42} \
    --n-splits 5 \
    --n-reps 10
