#!/bin/bash
# Evaluate on Barley: 48 nodes, 84 edges, 10000 samples.

python -u bench/run_eval.py \
    --network barley \
    --n_samples 10000 \
    --seed "${SEED:-42}"
