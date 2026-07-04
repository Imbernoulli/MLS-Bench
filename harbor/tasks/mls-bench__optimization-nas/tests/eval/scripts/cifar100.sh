#!/bin/bash
# Materialize this run's validation table (verifier-side), then run the search.
# ENV/SEED are exported BEFORE apply.py so the input generator materializes
# only the active run's table; MLSBENCH_EPHEMERAL_INPUTS=1 tells the runner it
# may delete the table after loading (apply.py re-creates it every evaluation).
export ENV=cifar100
export SEED="${SEED:-42}"
export MLSBENCH_EPHEMERAL_INPUTS=1
cd /workspace
python "/tests/eval/_inputgen/apply.py" "optimization-nas" /workspace
set -euo pipefail
cd naslib
NAS_EPOCHS=30 python custom_nas_search.py
