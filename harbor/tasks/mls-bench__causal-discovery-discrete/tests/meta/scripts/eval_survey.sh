#!/bin/bash
# Evaluate on Survey: 6 nodes, 6 edges, 500 samples.

python "/tests/eval/_inputgen/apply.py" "causal-discovery-discrete" /workspace
python -u bench/run_eval.py \
    --network survey \
    --n_samples 500 \
    --seed "${SEED:-42}"
