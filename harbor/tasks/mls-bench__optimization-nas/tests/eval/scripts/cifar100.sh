#!/bin/bash
# Materialize this run's validation table (verifier-side), then run the search.
# ENV/SEED are exported BEFORE apply.py so the input generator produces only
# the active run's table; MLSBENCH_EPHEMERAL_INPUTS=1 marks it ephemeral.
# The table is PIPED to the fixed wrapper as JSON (apply.py --emit-json ->
# fixed_entry.py --inputs-json-stdin): it NEVER touches the shared workspace
# filesystem, so a CONCURRENT sibling eval's agent code (Harbor runs the
# label x seed wave in parallel) has no on-disk staging window to read.
export ENV=cifar100
export SEED="${SEED:-42}"
export MLSBENCH_EPHEMERAL_INPUTS=1
cd /workspace
_EVAL_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_staged_list="$(mktemp /tmp/mlsb_staged.XXXXXX)"
# EXIT backstop: with --emit-json nothing is staged to disk (the list stays
# empty and this trap is a no-op); it still covers a version-skewed apply.py
# that stages the old way — such blobs are deleted here, and the wrapper's
# --inputs-glob pass unlinks any that remain before importing the module.
trap 'xargs -r rm -f -- < "$_staged_list" 2>/dev/null; rm -f "$_staged_list"' EXIT
set -euo pipefail
cd naslib
python "/tests/eval/_inputgen/apply.py" "optimization-nas" /workspace \
    --emit-json --list-out "$_staged_list" \
  | NAS_EPOCHS=30 python "$_EVAL_SCRIPTS_DIR/fixed_entry.py" \
      --module custom_nas_search.py \
      --inputs-json-stdin \
      --inputs-glob "naslib/data/nb201_tables_cifar100_s${SEED:-42}.json" \
      --entry _main --
