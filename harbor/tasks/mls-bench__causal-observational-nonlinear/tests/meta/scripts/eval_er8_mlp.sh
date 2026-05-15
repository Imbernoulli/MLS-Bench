#!/bin/bash
# Evaluate on ER8-MLP: Erdos-Renyi graph, 8 nodes, MLP nonlinearity, 500 samples, exponential noise.
# Working directory is /workspace (causal-learn package root).

python -u bench/run_eval.py \
    --graph_type er \
    --n_nodes 8 \
    --er_prob 0.3 \
    --n_samples 500 \
    --noise_type exp \
    --fn_type mlp \
    --seed "${SEED:-42}"
