#!/bin/bash
# Evaluate on Hailfinder: 56 nodes, 66 edges, 10000 samples (meteorology).

python -u bench/run_eval.py \
    --network hailfinder \
    --n_samples 10000 \
    --seed "${SEED:-42}"
