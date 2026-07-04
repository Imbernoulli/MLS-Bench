#!/bin/bash
# Evaluate on SF12: Scale-Free (BA) graph, 12 nodes, m=2, 300 samples, uniform noise.
# Uniform noise has zero kurtosis (weakest non-Gaussianity), making this the hardest scenario.
# Working directory is /workspace (causal-learn package root).

python -u bench/run_eval.py \
    --graph_type sf \
    --n_nodes 12 \
    --sf_m 2 \
    --n_samples 300 \
    --noise_type uniform \
    --seed "${SEED:-42}"
