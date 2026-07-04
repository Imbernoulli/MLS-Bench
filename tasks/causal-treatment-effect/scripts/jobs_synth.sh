#!/bin/bash
# Evaluate CATE estimator on an explicitly synthetic Jobs/LaLonde-inspired DGP.
cd /workspace
python scikit-learn/custom_cate.py \
    --dataset ddb3d2e9d0a6 \
    --seed ${SEED:-42} \
    --n-splits 5 \
    --n-reps 10
