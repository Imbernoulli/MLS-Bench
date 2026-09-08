#!/bin/bash
# Evaluate on ER10: Erdos-Renyi graph, 10 nodes, p=0.3, 500 samples.

python -u bench/run_eval.py \
    --graph_type er \
    --n_nodes 10 \
    --er_prob 0.3 \
    --n_samples 500 \
    --seed "${SEED:-42}"
