#!/bin/bash
# Evaluate on ER50: Erdos-Renyi graph, 50 nodes, p=0.2, 2000 samples, exponential noise.
# Large dense graph (~245 expected edges, avg ~4.9 parents/node).
# Working directory is /workspace (causal-learn package root).

python -u bench/run_eval.py \
    --graph_type er \
    --n_nodes 50 \
    --er_prob 0.2 \
    --n_samples 2000 \
    --noise_type exp \
    --seed "${SEED:-42}"
