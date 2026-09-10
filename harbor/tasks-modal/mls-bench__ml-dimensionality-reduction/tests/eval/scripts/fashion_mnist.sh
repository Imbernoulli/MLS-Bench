#!/bin/bash
# Evaluate custom dimensionality reduction on the Fashion-MNIST dataset.
cd /workspace

python -u scikit-learn/bench/custom_dimred.py \
    --dataset fashion_mnist \
    --seed "${SEED:-42}" \
    --n_neighbors 7
