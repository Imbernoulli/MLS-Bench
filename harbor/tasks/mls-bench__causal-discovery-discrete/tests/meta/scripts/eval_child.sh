#!/bin/bash
# Evaluate on Child: 20 nodes, 25 edges, 2000 samples (medical).

python -u bench/run_eval.py \
    --network child \
    --n_samples 2000 \
    --seed "${SEED:-42}"
