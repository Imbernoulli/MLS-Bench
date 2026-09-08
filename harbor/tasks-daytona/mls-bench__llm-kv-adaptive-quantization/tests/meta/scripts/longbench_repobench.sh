#!/bin/bash
set -euo pipefail
if [ -n "${TRANSFORMERS_KV_LAB_DIR:-}" ]; then
  cd "${TRANSFORMERS_KV_LAB_DIR}"
fi
mkdir -p "${OUTPUT_DIR:-./output}"
export HF_HOME="${HF_HOME:-${DATA_ROOT:-/data}/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-3B-Instruct}"
python custom_quant_eval.py \
  --workload longbench_repobench \
  --budget-bits 4 \
  --seed "${SEED:-42}"
