#!/bin/bash
# HumanEval benchmark with 4 agent nodes, MacNet backend: qwen2.5-72b-instruct
# Requires: QWEN_API_KEY env var (DashScope)
set -e

if [ -n "${MLSBENCH_PKG_DIR:-}" ] && [ -d "$MLSBENCH_PKG_DIR" ]; then
    cd "$MLSBENCH_PKG_DIR"
elif [ -d /workspace/chatdev-macnet ]; then
    cd /workspace/chatdev-macnet
fi

export MACNET_MODEL="qwen2.5-72b-instruct"
export OPENAI_API_KEY="${QWEN_API_KEY:-$OPENAI_API_KEY}"
export BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"

SEED=${SEED:-42}
export PYTHONHASHSEED=$SEED
export NODE_NUM=4

echo "=== Running HumanEval benchmark (backend: qwen2.5-72b-instruct, 4 nodes) ==="
echo "Seed: $SEED"

python3 run_humaneval.py --node_num 4 --seed "$SEED" --subset_step "${SUBSET_STEP:-5}" --macnet_timeout 600
