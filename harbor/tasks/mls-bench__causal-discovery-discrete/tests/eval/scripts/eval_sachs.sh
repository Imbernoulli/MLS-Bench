#!/bin/bash
# Evaluate on Sachs: 11 nodes, 17 edges, 1000 samples (protein signaling).

python -u bench/run_eval.py \
    --network sachs \
    --n_samples 1000 \
    --seed "${SEED:-42}"
