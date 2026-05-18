#!/bin/bash
# Evaluate on ER20 Dense: Erdos-Renyi graph, 20 nodes, p=0.5, 500 samples, laplace noise.
# Working directory is /workspace (causal-learn package root).

python -u bench/run_eval.py \
    --graph_type er \
    --n_nodes 20 \
    --er_prob 0.5 \
    --n_samples 500 \
    --noise_type laplace \
    --seed "${SEED:-42}"
