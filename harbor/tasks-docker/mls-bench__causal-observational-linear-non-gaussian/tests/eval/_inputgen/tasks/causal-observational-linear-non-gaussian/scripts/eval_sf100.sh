#!/bin/bash
# Evaluate on SF100: Scale-Free graph, 100 nodes, m=3, 1000 samples, uniform noise.
# Working directory is /workspace (causal-learn package root).

python -u bench/run_eval.py \
    --graph_type sf \
    --n_nodes 100 \
    --sf_m 3 \
    --n_samples 1000 \
    --noise_type uniform \
    --seed "${SEED:-42}"
