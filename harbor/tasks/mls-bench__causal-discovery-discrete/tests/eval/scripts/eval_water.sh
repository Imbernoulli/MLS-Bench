#!/bin/bash
# Evaluate on Water: 32 nodes, 66 edges, 5000 samples.

python -u bench/run_eval.py \
    --network water \
    --n_samples 5000 \
    --seed "${SEED:-42}"
