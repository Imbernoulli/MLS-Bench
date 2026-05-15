#!/bin/bash
# Evaluate CATE estimator on an explicitly synthetic Jobs/LaLonde-inspired DGP.
cd /workspace
python scikit-learn/custom_cate.py \
    --dataset jobs_synth \
    --seed ${SEED:-42} \
    --n-splits 5 \
    --n-reps 10
