#!/bin/bash
# Noisy variant: ER20-Hard (p=0.35, 400 samples) + higher noise (noise_scale=2.5).

python -u bench/run_eval.py \
    --graph_type er \
    --n_nodes 20 \
    --er_prob 0.35 \
    --n_samples 400 \
    --noise_scale 2.5 \
    --seed "${SEED:-42}"
