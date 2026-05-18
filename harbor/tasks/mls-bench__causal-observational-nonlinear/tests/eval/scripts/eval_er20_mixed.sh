#!/bin/bash
# Evaluate on ER20-Mixed: Erdos-Renyi graph, 20 nodes, mixed nonlinearity, 2000 samples, laplace noise.
# Working directory is /workspace (causal-learn package root).

python -u bench/run_eval.py \
    --graph_type er \
    --n_nodes 20 \
    --er_prob 0.3 \
    --n_samples 2000 \
    --noise_type laplace \
    --fn_type mixed \
    --seed "${SEED:-42}"
