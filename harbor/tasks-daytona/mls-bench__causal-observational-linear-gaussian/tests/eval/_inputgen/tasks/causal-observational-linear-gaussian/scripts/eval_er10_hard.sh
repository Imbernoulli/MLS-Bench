#!/bin/bash
# Hard variant: ER10 with denser graph (p=0.5) and fewer samples (200).

python -u bench/run_eval.py \
    --graph_type er \
    --n_nodes 10 \
    --er_prob 0.5 \
    --n_samples 200 \
    --seed "${SEED:-42}"
