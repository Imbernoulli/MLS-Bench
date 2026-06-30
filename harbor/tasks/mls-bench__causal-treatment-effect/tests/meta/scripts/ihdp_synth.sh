#!/bin/bash
# Evaluate CATE estimator on an explicitly synthetic IHDP-inspired DGP.
cd /workspace
python "/tests/eval/_inputgen/apply.py" "causal-treatment-effect" /workspace
python scikit-learn/custom_cate.py \
    --dataset 39d5166f6a43 \
    --seed ${SEED:-42} \
    --n-splits 5 \
    --n-reps 10
