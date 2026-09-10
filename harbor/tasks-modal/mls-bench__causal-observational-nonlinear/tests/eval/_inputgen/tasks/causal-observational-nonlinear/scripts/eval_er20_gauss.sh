#!/bin/bash
# Evaluate on ER20-Gauss: Erdos-Renyi graph, 20 nodes, mixed nonlinearity, 2000 samples, Gaussian noise.
# Tests with Gaussian noise where identifiability is harder (no non-Gaussianity to exploit).
# Working directory is /workspace (causal-learn package root).

python -u bench/run_eval.py \
    --graph_type er \
    --n_nodes 20 \
    --er_prob 0.3 \
    --n_samples 2000 \
    --noise_type gaussian \
    --fn_type mixed \
    --seed "${SEED:-42}"
