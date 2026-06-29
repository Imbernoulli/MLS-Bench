#!/bin/bash
# Evaluate on ER12-LowSample: Erdos-Renyi graph, 12 nodes, mixed nonlinearity, 150 samples, laplace noise.
# Tests performance in the low-sample regime.
# Working directory is /workspace (causal-learn package root).

python -u bench/run_eval.py \
    --graph_type er \
    --n_nodes 12 \
    --er_prob 0.3 \
    --n_samples 150 \
    --noise_type laplace \
    --fn_type mixed \
    --seed "${SEED:-42}"
