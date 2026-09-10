#!/bin/bash
# Noisy variant: SF50-Hard (m=3, 1000 samples) + higher noise (noise_scale=2.5).

python -u bench/run_eval.py \
    --graph_type sf \
    --n_nodes 50 \
    --sf_m 3 \
    --n_samples 1000 \
    --noise_scale 2.5 \
    --seed "${SEED:-42}"
