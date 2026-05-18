#!/bin/bash
# Evaluate on Cancer: 5 nodes, 4 edges, 500 samples.

python -u bench/run_eval.py \
    --network cancer \
    --n_samples 500 \
    --seed "${SEED:-42}"
