# Sourced by the eval scripts. Under Harbor the image keeps only the train and
# validation rows of each scored CSV (a `.mlsb_test_withheld` marker sits next
# to the truncated file) and the test rows arrive with tests/, mounted only at
# verification. mlsb_eval_root rebuilds the full CSV under $OUTPUT_DIR and
# prints the directory to pass as --root_path. Without the marker (native runs,
# where the image carries the full file) it prints the root it was given.
#
#   ROOT="$(mlsb_eval_root "$ROOT" "$DPATH")" || exit 1
mlsb_eval_root() {
  local root="${1%/}" name="$2" tail src manifest out
  # budget_check.py reads the scripts' argv by sourcing them with `python`
  # stubbed as a function; that pass needs no data, so do not rebuild any.
  if [ "$(type -t python)" = function ]; then printf '%s\n' "$1"; return 0; fi
  if [ ! -e "$root/.mlsb_test_withheld" ]; then
    printf '%s/\n' "$root"
    return 0
  fi
  case "$name" in
    ETTh1.csv) tail="ETTh1_tail.csv.gz" ;;
    weather.csv) tail="weather_tail.csv.gz" ;;
    electricity.csv) tail="electricity_tail.csv.gz" ;;
    *) echo "mlsb_eval_root: no withheld split is recorded for $name" >&2; return 1 ;;
  esac
  src="${MLSBENCH_TASK_DIR:-}/data/withheld/$tail"
  manifest="${MLSBENCH_TASK_DIR:-}/data/withheld/withheld_sha256.txt"
  if [ ! -f "$src" ] || [ ! -f "$manifest" ]; then
    echo "mlsb_eval_root: the withheld test split is not mounted ($src)" >&2
    return 1
  fi
  # The settings of one seed run concurrently and share $OUTPUT_DIR: key by label.
  out="${OUTPUT_DIR:-${TMPDIR:-/tmp}}/mlsb_eval_data/${ENV:-run}/seed${SEED:-42}/${name%.csv}"
  rm -rf "$out" && mkdir -p "$out" || return 1
  { cat "$root/$name" && gzip -dc "$src"; } > "$out/$name.part" || return 1
  mv -f "$out/$name.part" "$out/$name" || return 1
  if ! ( cd "$out" && grep "  $name\$" "$manifest" | sha256sum -c --quiet - ) >&2; then
    echo "mlsb_eval_root: rebuilt $name does not match the original data" >&2
    return 1
  fi
  printf '%s/\n' "$out"
}
