#!/bin/bash
# Hard variant: SF50 with denser graph (m=3) and fewer samples (1000).

python "/tests/eval/_inputgen/apply.py" "causal-observational-linear-gaussian" /workspace
python -u bench/run_eval.py \
    --graph_type sf \
    --n_nodes 50 \
    --sf_m 3 \
    --n_samples 1000 \
    --seed "${SEED:-42}"
