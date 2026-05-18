#!/bin/bash
# Noisy variant: ER10-Hard (p=0.5, 200 samples) + higher noise (noise_scale=2.5).

python -u bench/run_eval.py \
    --graph_type er \
    --n_nodes 10 \
    --er_prob 0.5 \
    --n_samples 200 \
    --noise_scale 2.5 \
    --seed "${SEED:-42}"
