#!/bin/bash
# Evaluate on ER15: Erdos-Renyi graph, 15 nodes, p=0.2, 500 samples, Laplace noise.
# Working directory is /workspace (causal-learn package root).

python -u bench/run_eval.py \
    --graph_type er \
    --n_nodes 15 \
    --er_prob 0.2 \
    --n_samples 500 \
    --noise_type laplace \
    --seed "${SEED:-42}"
