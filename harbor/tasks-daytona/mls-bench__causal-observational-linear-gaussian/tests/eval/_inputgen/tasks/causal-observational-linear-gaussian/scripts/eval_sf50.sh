#!/bin/bash
# Evaluate on SF50: Scale-Free graph, 50 nodes, m=2, 2000 samples.

python -u bench/run_eval.py \
    --graph_type sf \
    --n_nodes 50 \
    --sf_m 2 \
    --n_samples 2000 \
    --seed "${SEED:-42}"
