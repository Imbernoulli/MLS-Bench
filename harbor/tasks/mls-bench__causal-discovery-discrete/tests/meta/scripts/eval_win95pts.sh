#!/bin/bash
# Evaluate on Win95pts: 76 nodes, 112 edges, 10000 samples.

python -u bench/run_eval.py \
    --network win95pts \
    --n_samples 10000 \
    --seed "${SEED:-42}"
