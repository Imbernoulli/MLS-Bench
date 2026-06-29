#!/bin/bash
# Evaluate on ER20: Erdos-Renyi graph, 20 nodes, p=0.2, 1000 samples.

python -u bench/run_eval.py \
    --graph_type er \
    --n_nodes 20 \
    --er_prob 0.2 \
    --n_samples 1000 \
    --seed "${SEED:-42}"
