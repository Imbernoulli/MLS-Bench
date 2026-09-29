#!/bin/bash
# The harness re-stages this run's validation table before every evaluation,
# so it is treated as ephemeral (read+unlinked by the fixed wrapper before
# any editable code runs).
export MLSBENCH_EPHEMERAL_INPUTS=1
# Resolve the FIXED wrapper next to this script. It reads and unlinks the
# staged table BEFORE any editable code runs, keeps it in its own process,
# and runs the editable search in a separate process that can reach the
# table only through budgeted queries counted by the wrapper (-I: nothing
# from the workspace or PYTHON* variables is importable into it).
ORACLE_ENTRY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/nas_oracle_entry.py"
ENV=cifar100 NAS_EPOCHS=30 python -I -B "$ORACLE_ENTRY" \
  --module custom_nas_search.py \
  --inputs-glob "naslib/data/nb201_tables_cifar100_s${SEED:-42}.json" \
  --entry _main --
