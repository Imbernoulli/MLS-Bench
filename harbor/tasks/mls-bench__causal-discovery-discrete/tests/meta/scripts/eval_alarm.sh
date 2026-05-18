#!/bin/bash
# Evaluate on Alarm: 37 nodes, 46 edges, 5000 samples (medical monitoring).

python -u bench/run_eval.py \
    --network alarm \
    --n_samples 5000 \
    --seed "${SEED:-42}"
