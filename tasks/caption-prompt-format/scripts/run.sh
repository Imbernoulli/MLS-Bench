#!/usr/bin/env bash
set -euo pipefail

cd /workspace/image-captioning
task_root="${MLSBENCH_TASK_DIR:-${TASK_DIR:-/tests/meta}}"
private_data="${task_root}/data/image-captioning"
runtime_data="${OUTPUT_DIR:-/tmp}/caption-official-prompt-${SEED:-42}"
rm -rf "${runtime_data}"
mkdir -p "${runtime_data}"
trap 'rc=$?; rm -rf "${runtime_data}"; if [[ ${rc} -ne 0 ]]; then printf "VERIFICATION_FAILED image-captioning rc=%s\n" "${rc}" >&2; fi' EXIT
for name in source_manifest.json train_clip.pt train_refs.json eval_clip.pt eval_refs.json; do
    test -f "${private_data}/${name}"
    ln -s "${private_data}/${name}" "${runtime_data}/${name}"
done
test -f "${CAPTION_GPT2:-/data/image-captioning/gpt2}/config.json"

python harness.py \
    --mode prompt \
    --config solution/prompt.py \
    --data-root "${runtime_data}" \
    --gpt-dir "${CAPTION_GPT2:-/data/image-captioning/gpt2}" \
    --seed "${SEED:-42}"
