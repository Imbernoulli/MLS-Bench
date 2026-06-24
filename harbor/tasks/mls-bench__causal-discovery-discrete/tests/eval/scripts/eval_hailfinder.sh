#!/bin/bash
# Evaluate on Hailfinder: 56 nodes, 66 edges, 10000 samples (meteorology).

python "/tests/eval/_inputgen/apply.py" "causal-discovery-discrete" /workspace
python -u bench/run_eval.py \
    --network hailfinder \
    --n_samples 10000 \
    --seed "${SEED:-42}"
