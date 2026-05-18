#!/bin/bash
# Evaluate custom dimensionality reduction on the 20 Newsgroups (5 categories) dataset.
cd /workspace

python -u scikit-learn/bench/custom_dimred.py \
    --dataset newsgroups \
    --seed "${SEED:-42}" \
    --n_neighbors 7
