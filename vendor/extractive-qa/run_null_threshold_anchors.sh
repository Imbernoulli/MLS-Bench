#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_ROOT="${QA_ANCHOR_OUT:?QA_ANCHOR_OUT is required}"

declare -A BASELINE_FILES=(
  [threshold_neg30]="null_threshold__threshold_neg30.py"
  [threshold_0]="null_threshold__threshold_0.py"
  [threshold_30]="null_threshold__threshold_30.py"
)
declare -A DATASET_FILES=(
  [part0]="squad2_validation_part0.jsonl"
  [part1]="squad2_validation_part1.jsonl"
  [part2]="squad2_validation_part2.jsonl"
)

if [[ "$#" -eq 0 ]]; then
  echo "QA_ANCHOR_ERROR at least one baseline__partition key is required" >&2
  exit 2
fi

mkdir -p "$OUT_ROOT"
for key in "$@"; do
  baseline="${key%%__*}"
  partition="${key##*__}"
  if [[ -z "${BASELINE_FILES[$baseline]:-}" || -z "${DATASET_FILES[$partition]:-}" ]]; then
    echo "QA_ANCHOR_ERROR invalid key=$key" >&2
    exit 2
  fi
  run_dir="$OUT_ROOT/$baseline/$partition"
  mkdir -p "$run_dir"
  printf '125\n' >"$run_dir/rc"
  printf '%q ' python -u "$ROOT/harness_null_threshold.py" \
    --solution "$ROOT/baselines/${BASELINE_FILES[$baseline]}" \
    --dataset "${DATASET_FILES[$partition]}" --seed 42 \
    >"$run_dir/command.txt"
  printf '\n' >>"$run_dir/command.txt"
done

python - <<'PY'
import huggingface_hub
import importlib.metadata
import numpy
import safetensors
import tokenizers
import torch
import transformers

expected = {
    "numpy": (numpy.__version__, "2.1.2"),
    "transformers": (transformers.__version__, "4.49.0"),
    "tokenizers": (tokenizers.__version__, "0.21.0"),
    "huggingface_hub": (huggingface_hub.__version__, "0.28.1"),
    "safetensors": (safetensors.__version__, "0.5.2"),
    "regex": (importlib.metadata.version("regex"), "2024.11.6"),
}
mismatches = [
    f"{name}={actual} (expected {wanted})"
    for name, (actual, wanted) in expected.items()
    if actual != wanted
]
if mismatches:
    raise SystemExit("QA_ANCHOR_ERROR image dependency mismatch: " + ", ".join(mismatches))
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("QA_ANCHOR_ERROR CUDA is unavailable")
print(
    "QA_ANCHOR_ENV "
    f"gpus={torch.cuda.device_count()} "
    f"torch={torch.__version__} "
    + " ".join(f"{name}={actual}" for name, (actual, _) in expected.items()),
    flush=True,
)
for index in range(torch.cuda.device_count()):
    print(f"QA_ANCHOR_GPU index={index} name={torch.cuda.get_device_name(index)!r}", flush=True)
PY

gpu_count="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
if [[ "$gpu_count" -lt 1 ]]; then
  echo "QA_ANCHOR_ERROR no visible GPUs" >&2
  exit 3
fi

run_one() {
  local key="$1"
  local gpu="$2"
  local baseline="${key%%__*}"
  local partition="${key##*__}"
  local run_dir="$OUT_ROOT/$baseline/$partition"
  set +e
  CUDA_VISIBLE_DEVICES="$gpu" python -u "$ROOT/harness_null_threshold.py" \
    --solution "$ROOT/baselines/${BASELINE_FILES[$baseline]}" \
    --dataset "${DATASET_FILES[$partition]}" \
    --seed 42 >"$run_dir/worker.log" 2>&1
  local rc="$?"
  set -e
  printf '%s\n' "$rc" >"$run_dir/rc"
  printf 'QA_ANCHOR_RESULT key=%s gpu=%s rc=%s\n' "$key" "$gpu" "$rc"
}

keys=("$@")
offset=0
while [[ "$offset" -lt "${#keys[@]}" ]]; do
  pids=()
  wave_keys=()
  for ((gpu=0; gpu<gpu_count && offset<${#keys[@]}; gpu++, offset++)); do
    key="${keys[$offset]}"
    run_one "$key" "$gpu" &
    pids+=("$!")
    wave_keys+=("$key")
  done
  for index in "${!pids[@]}"; do
    wait "${pids[$index]}"
  done
done

failed=0
for key in "${keys[@]}"; do
  baseline="${key%%__*}"
  partition="${key##*__}"
  rc="$(tr -d '[:space:]' <"$OUT_ROOT/$baseline/$partition/rc")"
  if [[ "$rc" != "0" ]]; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "QA_ANCHOR_ERROR one or more workloads failed" >&2
  exit 1
fi
echo "QA_ANCHOR_ALL_DONE workloads=${#keys[@]}"
