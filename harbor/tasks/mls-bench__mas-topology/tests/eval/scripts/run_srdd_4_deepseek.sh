#!/bin/bash
# SRDD benchmark with 4 agent nodes, MacNet backend: deepseek-chat
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

SRDD_QUERY_PATH="${SRDD_QUERY_PATH:-srdd_queries.json}"

echo "=== Running SRDD benchmark (backend: deepseek-chat, 4 nodes) ==="
echo "Seed: $SEED"
echo "Query path: $SRDD_QUERY_PATH"

python3 run_srdd.py --node_num 4 --seed "$SEED" --query_path "$SRDD_QUERY_PATH" --subset_step "${SRDD_SUBSET_STEP:-1}" --macnet_timeout 600
