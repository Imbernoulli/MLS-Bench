#!/bin/bash
# Evaluate on Asia: 8 nodes, 8 edges, 1000 samples (medical/lung disease).

python "/tests/eval/_inputgen/apply.py" "causal-discovery-discrete" /workspace
python -u bench/run_eval.py \
    --network asia \
    --n_samples 1000 \
    --seed "${SEED:-42}"
