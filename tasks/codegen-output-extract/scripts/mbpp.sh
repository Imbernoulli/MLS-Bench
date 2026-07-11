#!/bin/bash
# Evaluate the editable extraction policy with a fixed prompt and decoder.
set -euo pipefail
: "${MLSBENCH_VERIFIER_DATA_ROOT:?verifier data root is required}"
private_data="${MLSBENCH_VERIFIER_DATA_ROOT}/code-generation"
runtime_data="$(mktemp -d "${TMPDIR:-/tmp}/codegen-private.XXXXXX")"
trap 'rm -rf -- "${runtime_data}"' EXIT
test -s "${private_data}/problems.json"
test -s "${private_data}/manifest.json"
cp -- "${private_data}/problems.json" "${private_data}/manifest.json" "${runtime_data}/"
chmod 0400 "${runtime_data}/problems.json" "${runtime_data}/manifest.json"
export CG_DATA="${runtime_data}" CG_PRIVATE_DATA_COPY=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd /workspace/code-generation-lab

python harness_extract.py \
    --solution solution/policy_extract.py \
    --seed ${SEED:-42} \
    --n 257
