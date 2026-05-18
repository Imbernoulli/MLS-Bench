#!/bin/bash
# Evaluate on ER12-GP: Erdos-Renyi graph, 12 nodes, GP nonlinearity, 1000 samples, laplace noise.
# Working directory is /workspace (causal-learn package root).

python -u bench/run_eval.py \
    --graph_type er \
    --n_nodes 12 \
    --er_prob 0.3 \
    --n_samples 1000 \
    --noise_type laplace \
    --fn_type gp \
    --seed "${SEED:-42}"
