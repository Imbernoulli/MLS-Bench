#!/bin/bash
# Evaluate on Mildew: 35 nodes, 46 edges, 5000 samples.

python "/tests/eval/_inputgen/apply.py" "causal-discovery-discrete" /workspace
python -u bench/run_eval.py \
    --network mildew \
    --n_samples 5000 \
    --seed "${SEED:-42}"
