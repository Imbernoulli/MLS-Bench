#!/bin/bash
# Evaluate on the Win95pts case (hidden).
# The dataset is pre-generated into the workspace at setup time; the network
# identity and ground truth are held out off-machine. The driver picks the
# pre-generated input by the ENV label and emits a CAUSAL_PRED line that the
# host-side scorer grades.

python -u bench/run_eval.py \
    --label "${ENV:-Win95pts}" \
    --seed "${SEED:-42}"
