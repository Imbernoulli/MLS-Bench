#!/bin/bash
# HumanEval benchmark with 4 agent nodes, MacNet backend: deepseek-chat
# Requires: DEEPSEEK_API_KEY env var
set -e

if [ -n "${MLSBENCH_PKG_DIR:-}" ] && [ -d "$MLSBENCH_PKG_DIR" ]; then
    cd "$MLSBENCH_PKG_DIR"
elif [ -d /workspace/chatdev-macnet ]; then
    cd /workspace/chatdev-macnet
fi

export MACNET_MODEL="deepseek-chat"
export OPENAI_API_KEY="${DEEPSEEK_API_KEY:-$OPENAI_API_KEY}"
export BASE_URL="https://api.deepseek.com/v1"

SEED=${SEED:-42}
export PYTHONHASHSEED=$SEED
export NODE_NUM=4

echo "=== Running HumanEval benchmark (backend: deepseek-chat, 4 nodes) ==="
echo "Seed: $SEED"

python3 run_humaneval.py --node_num 4 --seed "$SEED" --subset_step "${SUBSET_STEP:-5}" --macnet_timeout 600
