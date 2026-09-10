#!/bin/bash
# Core training/evaluation script for agent-tool-reasoning.
#
# Launchers (run_I1_*.sh) set AGENT_* and TOOL_SERVER_* env vars before
# exec'ing this script. Do NOT run this script directly — always go through
# a launcher so the backend is configured.
set -e

# cd into package dir (handles apptainer/docker/local workdir)
if [ -n "${MLSBENCH_PKG_DIR:-}" ] && [ -d "$MLSBENCH_PKG_DIR" ]; then
    cd "$MLSBENCH_PKG_DIR"
elif [ -d stabletoolbench ]; then
    cd stabletoolbench
fi
export PYTHONPATH=".:./toolbench/inference"

# TOOL_SERVER_DATA_DIR is injected by pkg_config in local mode; provide a
# sane default for legacy docker invocations.
TOOL_SERVER_DATA_DIR="${TOOL_SERVER_DATA_DIR:-/root/server_data}"

# ── Per-run configuration ────────────────────────────────────────────
SEED="${SEED:-42}"
export PYTHONHASHSEED="$SEED"

# Each test_cmd invocation gets its own timestamped output subdir so that
# multiple test rounds within a single agent run do NOT overwrite each
# other (qa_pipeline's --overwrite wipes output_answer_file at start).
# The TEST_TS is embedded in the TEST_METRICS line so the parser can
# correlate the leaderboard row back to the exact answer files (needed
# for per-row post-hoc SoPR computation).
LABEL="${ENV:-I1-instruction}"
OUTPUT_DIR="${OUTPUT_DIR:-./results}"
TEST_TS="${TEST_TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
export TEST_TS
SETTING_OUT="${OUTPUT_DIR}/${LABEL}/${TEST_TS}"
mkdir -p "$SETTING_OUT"

# Pick a random unused port so parallel jobs don't collide on the cache server.
if [ -z "${SERVER_PORT:-}" ]; then
    SERVER_PORT="$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()' 2>/dev/null || echo $((18080 + RANDOM % 1000)))"
fi
export SERVICE_URL="http://localhost:${SERVER_PORT}/virtual"

# ── Backend configuration (expected to be set by launcher) ────────────
# AGENT_*:       model under evaluation (varies per setting)
# TOOL_SERVER_*: simulates RapidAPI tool responses (fixed at deepseek-chat
#                via DeepSeek official API across all settings, for
#                reproducibility of tool responses).
: "${AGENT_MODEL:?AGENT_MODEL not set — invoke via a launcher script (run_I1_*.sh)}"
: "${AGENT_BASE_URL:?AGENT_BASE_URL not set — invoke via a launcher script}"
: "${AGENT_KEY:?AGENT_KEY not set — invoke via a launcher script}"
: "${TOOL_SERVER_MODEL:?TOOL_SERVER_MODEL not set — invoke via a launcher script}"
: "${TOOL_SERVER_BASE_URL:?TOOL_SERVER_BASE_URL not set — invoke via a launcher script}"
: "${TOOL_SERVER_KEY:?TOOL_SERVER_KEY not set — invoke via a launcher script}"

echo "Setting      : ${LABEL}"
echo "Test TS      : ${TEST_TS}"
echo "Cache server : model=${TOOL_SERVER_MODEL} base=${TOOL_SERVER_BASE_URL}"
echo "Agent        : model=${AGENT_MODEL} base=${AGENT_BASE_URL}"
echo "Output dir   : ${SETTING_OUT}"

# ── Step 1: Configure and start the cache server ──────────────────────
cd server

cat > config.yml <<YAML
api_key: "${TOOL_SERVER_KEY}"
api_base: "${TOOL_SERVER_BASE_URL}"
model: "${TOOL_SERVER_MODEL}"
temperature: 0
toolbench_url: "http://8.130.32.149:8080/rapidapi"
tools_folder: "${TOOL_SERVER_DATA_DIR}/tools"
cache_folder: "${TOOL_SERVER_DATA_DIR}/tool_response_cache"
# Disable cache writes during evaluation: multiple parallel jobs share this
# directory, and concurrent writes can corrupt cache files. The shipped cache
# already covers all queries used by the benchmark.
is_save: false
port: ${SERVER_PORT}
log_file: "./server.log"
YAML

python main.py &
SERVER_PID=$!
cd ..

echo "Waiting for cache server on port ${SERVER_PORT}..."
for i in $(seq 1 30); do
    if curl -s "http://localhost:${SERVER_PORT}/docs" > /dev/null 2>&1; then
        echo "Server ready."
        break
    fi
    sleep 2
done

# ── Step 2: Run inference ─────────────────────────────────────────────
COMMON_ARGS=(
    --backbone_model chatgpt_function
    --chatgpt_model "${AGENT_MODEL}"
    --base_url "${AGENT_BASE_URL}"
    --openai_key "${AGENT_KEY}"
    --tool_root_dir "${TOOL_SERVER_DATA_DIR}/tools"
    --method CustomSearch
    --toolbench_key ""
    --max_observation_length 1024
    --single_chain_max_step 12
    --max_query_count 60
    --num_thread 1
    --overwrite
)

# We evaluate on a fixed 50-query subset of StableToolBench's I1-instruction
# split (shipped as tasks/agent-tool-reasoning/scripts/test_50q.json). The
# full 163-query run was too expensive at ~100+ h per agent given
# max_tests=3 and 3 settings.
QUERY_FILE="${MLSBENCH_TASK_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/scripts/test_50q.json"
if [ ! -f "${QUERY_FILE}" ]; then
    echo "ERROR: query file not found: ${QUERY_FILE}" >&2
    exit 1
fi
echo "=== Running inference (label=${LABEL}, queries=${QUERY_FILE}) ==="
python toolbench/inference/qa_pipeline_multithread.py \
    "${COMMON_ARGS[@]}" \
    --input_query_file "${QUERY_FILE}" \
    --output_answer_file "${SETTING_OUT}/G1_instruction" || true

# ── Step 3: Calculate metrics ─────────────────────────────────────────
echo "=== Calculating metrics ==="
SETTING_OUT="${SETTING_OUT}" python3 << 'PYEOF'
import os, json, sys

def compute_metrics(result_dir):
    total = passed = total_queries = gave_up = 0
    if not os.path.isdir(result_dir):
        print(f"WARNING: {result_dir} not found", file=sys.stderr)
        return None
    for f in sorted(os.listdir(result_dir)):
        if not f.endswith('.json'):
            continue
        total += 1
        with open(os.path.join(result_dir, f)) as fh:
            data = json.load(fh)
        if data.get('win', False):
            passed += 1
        ag = data.get('answer_generation', {})
        total_queries += ag.get('query_count', 0)
        if ag.get('finish_type', '') == 'give_up':
            gave_up += 1
    if total == 0:
        return None
    return {
        'total': total,
        'passed': passed,
        'pass_rate': passed / total,
        'avg_queries': total_queries / total,
        'give_up_rate': gave_up / total,
    }

setting_out = os.environ['SETTING_OUT']
test_ts = os.environ.get('TEST_TS', '')
m = compute_metrics(os.path.join(setting_out, 'G1_instruction'))
if m:
    # answer_ts lets downstream tools (e.g. compute_sopr) locate the exact
    # answer-file directory that produced these metrics, even across many
    # test rounds that share a workspace.
    print(f'TEST_METRICS: pass_rate={m["pass_rate"]:.4f} avg_queries={m["avg_queries"]:.1f} give_up_rate={m["give_up_rate"]:.4f} answer_ts={test_ts}', flush=True)
else:
    print('ERROR: no inference results found', file=sys.stderr)
    sys.exit(1)
PYEOF

# ── Cleanup ───────────────────────────────────────────────────────────
kill $SERVER_PID 2>/dev/null || true
