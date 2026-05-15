#!/bin/bash
# Launcher: agent = deepseek-chat via DeepSeek official API.
# MLS-Bench only copies the file named by test_cmds[].cmd into the workspace;
# helper scripts (_common.sh, train.sh) stay in the source task dir and are
# reached via $MLSBENCH_TASK_DIR.
export AGENT_MODEL="deepseek-chat"
export AGENT_BASE_URL="https://api.deepseek.com/v1"
export AGENT_PROVIDER="deepseek"

TASK_DIR="${MLSBENCH_TASK_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
exec bash "${TASK_DIR}/scripts/_common.sh"
