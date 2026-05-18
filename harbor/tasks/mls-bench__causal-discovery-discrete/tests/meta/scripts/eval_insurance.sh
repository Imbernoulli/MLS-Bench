#!/bin/bash
# Evaluate on Insurance: 27 nodes, 52 edges, 5000 samples (automotive insurance).

python -u bench/run_eval.py \
    --network insurance \
    --n_samples 5000 \
    --seed "${SEED:-42}"
