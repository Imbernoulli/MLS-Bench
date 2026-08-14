#!/bin/bash
# The harness re-stages this run's validation table before every evaluation,
# so it is treated as ephemeral (read+unlinked by the fixed wrapper before
# any editable code runs).
export MLSBENCH_EPHEMERAL_INPUTS=1
# Resolve the FIXED wrapper next to this script; it preloads and unlinks the
# staged table BEFORE importing the editable module, so top-level statements
# in the editable range never see it on disk.
FIXED_ENTRY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/fixed_entry.py"
ENV=cifar100 NAS_EPOCHS=30 python "$FIXED_ENTRY" \
  --module custom_nas_search.py \
  --inputs-glob "naslib/data/nb201_tables_cifar100_s${SEED:-42}.json" \
  --entry _main --
