#!/bin/bash
# Hard variant: ER20 with denser graph (p=0.35) and fewer samples (400).

python -u bench/run_eval.py \
    --graph_type er \
    --n_nodes 20 \
    --er_prob 0.35 \
    --n_samples 400 \
    --seed "${SEED:-42}"
