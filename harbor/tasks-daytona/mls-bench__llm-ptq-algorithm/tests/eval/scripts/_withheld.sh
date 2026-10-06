# Sourced by the eval scripts. Under Harbor the image ships the WikiText-2 cache
# without its test split (a `.mlsb_test_withheld` marker sits in the cache dir)
# and the split arrives with tests/, mounted only at verification.
# mlsb_eval_wikitext rebuilds a complete cache under $OUTPUT_DIR and prints its
# path for HF_DATASETS_CACHE. Without the marker (native runs, where the image
# carries the full cache) it prints the cache it was given.
#
#   export HF_DATASETS_CACHE="$(mlsb_eval_wikitext /data/wikitext2)" || exit 1
mlsb_eval_wikitext() {
  local cache="${1%/}" src manifest out cfg
  # budget_check.py reads the scripts' argv by sourcing them with `python`
  # stubbed as a function; that pass needs no data, so do not rebuild any.
  if [ "$(type -t python)" = function ]; then printf '%s\n' "$1"; return 0; fi
  if [ ! -e "$cache/.mlsb_test_withheld" ]; then
    printf '%s\n' "$cache"
    return 0
  fi
  src="${MLSBENCH_TASK_DIR:-}/data/withheld/wikitext-test.arrow"
  manifest="${MLSBENCH_TASK_DIR:-}/data/withheld/withheld_sha256.txt"
  if [ ! -f "$src" ] || [ ! -f "$manifest" ]; then
    echo "mlsb_eval_wikitext: the withheld test split is not mounted ($src)" >&2
    return 1
  fi
  # The settings of one seed run concurrently and share $OUTPUT_DIR: key by label.
  out="${OUTPUT_DIR:-${TMPDIR:-/tmp}}/mlsb_eval_data/${ENV:-run}/seed${SEED:-42}/wikitext2"
  cfg="wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3"
  rm -rf "$out" && mkdir -p "$(dirname "$out")" || return 1
  cp -r "$cache" "$out" && rm -f "$out/.mlsb_test_withheld" && cp "$src" "$out/$cfg/wikitext-test.arrow" || return 1
  if ! ( cd "$out/$cfg" && sha256sum -c --quiet "$manifest" ) >&2; then
    echo "mlsb_eval_wikitext: restored wikitext-test.arrow does not match the original" >&2
    return 1
  fi
  printf '%s\n' "$out"
}
