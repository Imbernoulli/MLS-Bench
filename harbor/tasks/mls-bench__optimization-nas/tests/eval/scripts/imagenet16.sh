#!/bin/bash
# Materialize this run's validation table (verifier-side), then run the search.
# ENV/SEED are exported BEFORE apply.py so the input generator materializes
# only the active run's table; MLSBENCH_EPHEMERAL_INPUTS=1 marks it ephemeral
# (the fixed wrapper reads+unlinks it before any editable code runs; apply.py
# re-creates it every evaluation).
export ENV=imagenet16
export SEED="${SEED:-42}"
export MLSBENCH_EPHEMERAL_INPUTS=1
cd /workspace
_EVAL_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Splice the fixed runner blocks into stale baked workspace images (anchored
# exact-match replaces; no-op when the workspace is already up to date).
source "$_EVAL_SCRIPTS_DIR/_runtime_patch.sh"
_staged_list="$(mktemp /tmp/mlsb_staged.XXXXXX)"
# EXIT backstop: delete exactly the blobs apply.py staged for THIS eval,
# even when the runner crashes/times out before its in-process scrub —
# a crashed eval must not leave its withheld inputs readable for the rest
# of the wave. Normally a no-op (the fixed wrapper unlinks right after
# preloading).
trap 'xargs -r rm -f -- < "$_staged_list" 2>/dev/null; rm -f "$_staged_list"' EXIT
python "/tests/eval/_inputgen/apply.py" "optimization-nas" /workspace --list-out "$_staged_list"
set -euo pipefail
cd naslib
NAS_EPOCHS=30 python "$_EVAL_SCRIPTS_DIR/fixed_entry.py" \
  --module custom_nas_search.py \
  --inputs-glob "naslib/data/nb201_tables_*.json" \
  --entry _main --
