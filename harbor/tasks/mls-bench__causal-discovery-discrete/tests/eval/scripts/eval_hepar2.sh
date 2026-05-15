#!/bin/bash
# Evaluate on Hepar2: 70 nodes, 123 edges, 10000 samples.

python -u bench/run_eval.py \
    --network hepar2 \
    --n_samples 10000 \
    --seed "${SEED:-42}"
