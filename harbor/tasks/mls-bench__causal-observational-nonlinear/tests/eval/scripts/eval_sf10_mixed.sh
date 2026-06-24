#!/bin/bash
# Evaluate on SF10-Mixed: Scale-Free graph, 10 nodes, mixed nonlinearity, 500 samples, uniform noise.
# Working directory is /workspace (causal-learn package root).

python "/tests/eval/_inputgen/apply.py" "causal-observational-nonlinear" /workspace
python -u bench/run_eval.py \
    --graph_type sf \
    --n_nodes 10 \
    --n_samples 500 \
    --noise_type uniform \
    --fn_type mixed \
    --seed "${SEED:-42}"
