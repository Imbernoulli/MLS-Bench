#!/bin/bash
# Evaluate on SF20-GP: Scale-Free graph, 20 nodes, GP nonlinearity, 2000 samples, exp noise.
# Working directory is /workspace (causal-learn package root).

python -u bench/run_eval.py \
    --graph_type sf \
    --n_nodes 20 \
    --n_samples 2000 \
    --noise_type exp \
    --fn_type gp \
    --seed "${SEED:-42}"
