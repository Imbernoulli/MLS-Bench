#!/bin/bash
# Evaluate on ER15-Sigmoid: Erdos-Renyi graph, 15 nodes, sigmoid nonlinearity, 1000 samples, exp noise.
# Working directory is /workspace (causal-learn package root).

python -u bench/run_eval.py \
    --graph_type er \
    --n_nodes 15 \
    --er_prob 0.3 \
    --n_samples 1000 \
    --noise_type exp \
    --fn_type sigmoid \
    --seed "${SEED:-42}"
