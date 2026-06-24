#!/bin/bash
# Evaluate on Barley: 48 nodes, 84 edges, 10000 samples.

python "/tests/eval/_inputgen/apply.py" "causal-discovery-discrete" /workspace
python -u bench/run_eval.py \
    --network barley \
    --n_samples 10000 \
    --seed "${SEED:-42}"
