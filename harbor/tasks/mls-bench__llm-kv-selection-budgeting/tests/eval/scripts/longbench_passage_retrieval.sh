#!/usr/bin/env bash
# LongBench passage-retrieval workload.
set -euo pipefail
cd /workspace/transformers-kv-lab
mkdir -p "${OUTPUT_DIR:-./output}"
export MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-3B-Instruct}"
python custom_selection_eval.py \
  --workload longbench_passage_retrieval \
  --model-id "${MODEL_ID}" \
  --compression-ratio "${SELECTION_KV_COMPRESSION_RATIO:-0.8}" \
  --max-examples "${SELECTION_KV_MAX_EXAMPLES:-0}" \
  --seed "${SEED:-42}"
