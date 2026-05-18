#!/bin/bash
# Shared env setup for agent-tool-reasoning launcher scripts.
# Source (not exec) this from each run_I1_*.sh after defining:
#   AGENT_MODEL, AGENT_BASE_URL, AGENT_PROVIDER ("deepseek" or "dashscope")

set -e

# Resolve task dir (for fallback key files).
TASK_DIR="${MLSBENCH_TASK_DIR:-}"
if [ -z "$TASK_DIR" ]; then
    if [ -d "/workspace/_task" ]; then
        TASK_DIR="/workspace/_task"
    else
        TASK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    fi
fi

# Read a key: prefer env var, fall back to a file under task dir.
# Args: env_var_name fallback_filename
read_key() {
    local var="$1"
    local fname="$2"
    local val="${!var:-}"
    if [ -z "$val" ]; then
        for keyfile in "${TASK_DIR}/${fname}" "/workspace/_task/${fname}"; do
            if [ -n "$keyfile" ] && [ -f "$keyfile" ]; then
                val="$(tr -d '\n' < "$keyfile")"
                break
            fi
        done
    fi
    if [ -z "$val" ]; then
        echo "ERROR: ${var} is not set and ${fname} not found under ${TASK_DIR}" >&2
        return 1
    fi
    echo "$val"
}

# Agent key depends on provider; tool-server defaults to DeepSeek for
# reproducibility but every TOOL_SERVER_* var is overridable from the env
# so smoke tests with only one API key (e.g. only DASHSCOPE_API_KEY) can
# point both agent and tool-server at the same provider.
case "${AGENT_PROVIDER:-}" in
    deepseek)
        DEEPSEEK_API_KEY="$(read_key DEEPSEEK_API_KEY .deepseek_key)" || exit 1
        export AGENT_KEY="${DEEPSEEK_API_KEY}"
        ;;
    dashscope)
        DASHSCOPE_API_KEY="$(read_key DASHSCOPE_API_KEY .dashscope_key)" || exit 1
        export AGENT_KEY="${DASHSCOPE_API_KEY}"
        # Optional fallback: only require DEEPSEEK_API_KEY when the tool
        # server actually points at deepseek (the default below).
        DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-${DASHSCOPE_API_KEY}}"
        ;;
    *)
        echo "ERROR: AGENT_PROVIDER must be 'deepseek' or 'dashscope' (got '${AGENT_PROVIDER:-}')" >&2
        exit 1
        ;;
esac

# Tool server defaults: DeepSeek official deepseek-chat. Override any of
# TOOL_SERVER_MODEL / TOOL_SERVER_BASE_URL / TOOL_SERVER_KEY from the
# caller's environment (e.g. set TOOL_SERVER_BASE_URL to dashscope when
# only a Qwen key is available).
export TOOL_SERVER_MODEL="${TOOL_SERVER_MODEL:-deepseek-chat}"
export TOOL_SERVER_BASE_URL="${TOOL_SERVER_BASE_URL:-https://api.deepseek.com/v1}"
export TOOL_SERVER_KEY="${TOOL_SERVER_KEY:-${DEEPSEEK_API_KEY}}"

# Locate train.sh (next to this file, or inside the mounted task dir).
if [ -n "${MLSBENCH_TASK_DIR:-}" ] && [ -f "${MLSBENCH_TASK_DIR}/scripts/train.sh" ]; then
    TRAIN_SH="${MLSBENCH_TASK_DIR}/scripts/train.sh"
else
    TRAIN_SH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/train.sh"
fi
exec bash "${TRAIN_SH}"
