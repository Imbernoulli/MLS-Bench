#!/bin/bash
# Evaluate on ER10: Erdos-Renyi graph, 10 nodes, p=0.3, 250 samples, exponential noise.
# Working directory is /workspace (causal-learn package root).

python -u bench/run_eval.py \
    --graph_type er \
    --n_nodes 10 \
    --er_prob 0.3 \
    --n_samples 250 \
    --noise_type exp \
    --seed "${SEED:-42}"
