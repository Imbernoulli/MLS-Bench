#!/bin/bash
# Evaluate on ER30: Erdos-Renyi graph, 30 nodes, p=0.25, 1000 samples, Laplace noise.
# Dense graph (~109 expected edges, avg ~3.6 parents/node) with limited samples.
# Working directory is /workspace (causal-learn package root).

python -u bench/run_eval.py \
    --graph_type er \
    --n_nodes 30 \
    --er_prob 0.25 \
    --n_samples 1000 \
    --noise_type laplace \
    --seed "${SEED:-42}"
