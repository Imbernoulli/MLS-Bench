#!/bin/bash
# Launcher: agent = qwen2.5-7b-instruct via Dashscope (Aliyun) OpenAI-compatible API.
# Helper scripts (_common.sh, train.sh) live in $MLSBENCH_TASK_DIR since
# MLS-Bench only copies the launcher itself into the workspace.
export AGENT_MODEL="qwen2.5-7b-instruct"
export AGENT_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export AGENT_PROVIDER="dashscope"

TASK_DIR="${MLSBENCH_TASK_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
exec bash "${TASK_DIR}/scripts/_common.sh"
