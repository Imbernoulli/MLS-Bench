#!/bin/bash
# Evaluate custom dimensionality reduction on the MNIST dataset.
cd /workspace

python -u scikit-learn/bench/custom_dimred.py \
    --dataset mnist \
    --seed "${SEED:-42}" \
    --n_neighbors 7
