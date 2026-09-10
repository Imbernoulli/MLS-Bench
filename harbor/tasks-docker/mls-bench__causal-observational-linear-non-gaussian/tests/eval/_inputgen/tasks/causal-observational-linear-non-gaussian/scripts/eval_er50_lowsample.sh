#!/bin/bash
# Evaluate on ER50 with low samples: Erdos-Renyi graph, 50 nodes, p=0.1, 250 samples, exponential noise.
# Working directory is /workspace (causal-learn package root).

python -u bench/run_eval.py \
    --graph_type er \
    --n_nodes 50 \
    --er_prob 0.1 \
    --n_samples 250 \
    --noise_type exp \
    --seed "${SEED:-42}"
