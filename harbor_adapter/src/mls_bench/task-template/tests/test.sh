#!/bin/bash
# Harbor verifier for an MLS-Bench task.
#
# Harbor mounts this directory at /tests/ only at verification time, so
# anything here is hidden from the agent during its work session. Layout
# expected inside /tests/:
#   /tests/test.sh                 (this script)
#   /tests/score_task.py           (the guard/run-evals/score helper)
#   /tests/meta/config.json
#   /tests/meta/parser.py
#   /tests/meta/score_spec.py
#   /tests/meta/leaderboard.csv
#   /tests/meta/[budget_check.py]
#   /tests/meta/pristine/<rel>     (declared-file pristines for byte-segment diff)
#   /tests/meta/pristine_manifest.json   (sha256 of every file under a guarded prefix)
#   /tests/eval/scripts/*.sh       (every configured eval script)

# Reset PATH so an agent-left python/pip shim under /workspace can't shadow
# the system interpreter the verifier uses. Strip every env var Python
# inspects during startup so an agent-planted PYTHONSTARTUP /
# PYTHONUSERBASE / sitecustomize.py won't be imported.
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE \
      PYTHONNOUSERSITE PYTHONIOENCODING PYTHONHASHSEED \
      LD_PRELOAD LD_LIBRARY_PATH LD_AUDIT
export PYTHONNOUSERSITE=1
# Prefer the python that owns the package's ML stack (numpy / torch / pandas
# / etc.) — that's the one budget_check.py and mlsbench.scoring need. The
# pytorch-based base images ship conda at /opt/conda; the system /usr/bin
# python is usually a bare Debian python without scientific deps.
#
# We still pass -I (isolated mode) on every invocation so the agent's
# planted PYTHON* envs, user site, and current directory are not on
# sys.path — `-I` does NOT disable system site-packages, so /opt/conda's
# numpy/torch remain importable.
for candidate in \
        /opt/conda/bin/python3 \
        /opt/conda/bin/python \
        /opt/miniconda3/bin/python3 \
        /opt/miniconda3/bin/python \
        /usr/local/bin/python3 \
        /usr/bin/python3; do
    if [ -x "${candidate}" ]; then
        PYTHON_BIN="${candidate}"
        break
    fi
done
if [ -z "${PYTHON_BIN:-}" ]; then
    PYTHON_BIN=$(command -v python3 2>/dev/null || command -v python)
fi
# Add the chosen interpreter's bin dir to PATH so child processes spawned by
# eval scripts inherit it instead of /usr/bin/python.
case ":$PATH:" in
    *:"$(dirname "${PYTHON_BIN}")":*) :;;
    *) export PATH="$(dirname "${PYTHON_BIN}"):${PATH}";;
esac
export MLSBENCH_VERIFIER_PYTHON="${PYTHON_BIN}"

set -uo pipefail

# Normalize the platform log directory before touching any prior artifacts.
# The verifier runs as root; an agent-created symlink/file must never redirect
# reward publication or preserve an old positive score.
if [ ! -d /logs ] || [ -L /logs ]; then
    echo "verifier log root is not a real directory" >&2
    exit 0
fi
if [ -L /logs/verifier ] || { [ -e /logs/verifier ] && [ ! -d /logs/verifier ]; }; then
    rm -f -- /logs/verifier || exit 0
fi
if ! mkdir -p /logs/verifier \
        || ! chown 0:0 /logs/verifier \
        || ! chmod 700 /logs/verifier; then
    echo "verifier log path is not a real directory" >&2
    exit 0
fi
# Retries must not inherit a prior attempt's score inputs.
if ! find /logs/verifier -mindepth 1 -depth -delete; then
    printf '0\n' > /logs/verifier/reward.txt 2>/dev/null || true
    echo "failed to clear stale verifier artifacts" >&2
    exit 0
fi
# Pre-write reward so Mangrove always finds a file even if we get killed.
if ! printf '0\n' > /logs/verifier/reward.txt; then
    echo "failed to initialize zero reward" >&2
    exit 0
fi
_CANDIDATE_REWARD="/logs/verifier/.reward.candidate"

# Snapshot the verifier python's integrity for postmortem auditing. This does
# not authorize a score and therefore must not interfere with the zero default.
"${PYTHON_BIN}" -I -c "import hashlib,sys; print('verifier_python_sha256='+hashlib.sha256(open(sys.executable,'rb').read()).hexdigest())" \
    > /logs/verifier/python_audit.txt 2>&1 || {
        printf '0\n' > /logs/verifier/reward.txt
        echo "failed to audit verifier interpreter" >&2
        exit 0
    }

TASK_ID="$(cat /tests/meta/task_id 2>/dev/null || echo unknown)"
PKG_NAME="$(cat /tests/meta/package 2>/dev/null || echo unknown)"
WORKDIR="$(cat /tests/meta/workdir 2>/dev/null || echo /workspace)"

PRIVATE_ROOT="$(mktemp -d /tmp/mlsbench-verifier.XXXXXX)" || {
    printf '0\n' > /logs/verifier/reward.txt
    exit 0
}
PRIVATE_META="${PRIVATE_ROOT}/meta"
EVAL_META="$(mktemp -d /tmp/mlsbench-eval-runtime.XXXXXX)" || {
    rm -rf "${PRIVATE_ROOT}"
    printf '0\n' > /logs/verifier/reward.txt
    exit 0
}
EVAL_ARTIFACT_ROOT="$(mktemp -d /tmp/mlsbench-eval-artifacts.XXXXXX)" || {
    rm -rf "${PRIVATE_ROOT}" "${EVAL_META}"
    printf '0\n' > /logs/verifier/reward.txt
    exit 0
}

_VERIFICATION_COMMITTED=0
_remove_reward_candidate() {
    if [ -n "${_CANDIDATE_REWARD:-}" ]; then
        rm -f -- "${_CANDIDATE_REWARD}" 2>/dev/null || true
    fi
}

_cleanup_verifier() {
    # Until the success proof has been checked, every normal exit and caught
    # signal must leave an explicit zero. This also covers termination after
    # score_task has published a candidate reward but before test.sh has
    # validated the proof.
    if [ "${_VERIFICATION_COMMITTED:-0}" -ne 1 ]; then
        printf '0\n' > /logs/verifier/reward.txt 2>/dev/null || true
    fi
    _remove_reward_candidate
    if [ -n "${_HB_PID:-}" ]; then
        kill "${_HB_PID}" 2>/dev/null || true
        wait "${_HB_PID}" 2>/dev/null || true
        _HB_PID=""
    fi
    if [ -n "${_LOG_STREAM_PID:-}" ]; then
        kill "${_LOG_STREAM_PID}" 2>/dev/null || true
        wait "${_LOG_STREAM_PID}" 2>/dev/null || true
        _LOG_STREAM_PID=""
    fi
    rm -rf "${PRIVATE_ROOT}" "${EVAL_META}" "${EVAL_ARTIFACT_ROOT}"
}

_abort_verifier() {
    _VERIFICATION_COMMITTED=0
    _cleanup_verifier
    exit 0
}

trap _cleanup_verifier EXIT
trap _abort_verifier HUP INT TERM

if ! cp -a /tests/meta "${PRIVATE_META}" \
        || ! cp /tests/score_task.py "${PRIVATE_ROOT}/score_task.py"; then
    printf '0\n' > /logs/verifier/reward.txt
    echo "failed to stage private verifier assets" \
        > /logs/verifier/score_error.txt
    exit 0
fi
for runtime_asset in \
        config.json \
        task_description.md \
        task_id \
        package \
        workdir \
        package_envs.json \
        gpu_count \
        gpu_compute_cap \
        scripts \
        data \
        third_party; do
    if [ -e "${PRIVATE_META}/${runtime_asset}" ]; then
        if ! cp -a "${PRIVATE_META}/${runtime_asset}" "${EVAL_META}/${runtime_asset}"; then
            printf '0\n' > /logs/verifier/reward.txt
            echo "failed to stage sanitized runtime metadata" \
                > /logs/verifier/score_error.txt
            exit 0
        fi
    fi
done
if ! chmod -R a-w /tests/meta /tests/mlsbench_src /tests/score_task.py \
        || ! chmod -R go-rwx /tests/meta /tests/mlsbench_src /tests/score_task.py \
        || ! chmod -R a-w "${PRIVATE_META}" "${PRIVATE_ROOT}/score_task.py" \
        || ! chmod go-rwx "${PRIVATE_ROOT}" \
        || ! chmod -R a+rX,go-w "${EVAL_META}"; then
    printf '0\n' > /logs/verifier/reward.txt
    echo "failed to secure verifier metadata" \
        > /logs/verifier/score_error.txt
    exit 0
fi

# Evaluation executes agent-controlled code. Run it without root privileges
# and make the submitted workspace read-only for that uid. The scorer and its
# metadata remain under PRIVATE_ROOT (0700) and are never exposed through an
# environment variable or the legacy /workspace/_task link.
export MLSBENCH_EVAL_UID="${MLSBENCH_EVAL_UID:-65534}"
export MLSBENCH_EVAL_GID="${MLSBENCH_EVAL_GID:-65534}"
export MLSBENCH_CLEAN_PROCESS_GROUPS=1
export MLSBENCH_EVAL_ARTIFACT_ROOT="${EVAL_ARTIFACT_ROOT}"
if [ "$(id -u)" -eq 0 ]; then
    if ! chown "${MLSBENCH_EVAL_UID}:${MLSBENCH_EVAL_GID}" "${EVAL_ARTIFACT_ROOT}" \
            || ! chmod 700 "${EVAL_ARTIFACT_ROOT}" \
            || ! chown -R 0:0 "${WORKDIR}" 2>/dev/null \
            || ! chmod -R a-s,go-w,a+rX "${WORKDIR}" 2>/dev/null; then
        echo "0" > /logs/verifier/reward.txt
        echo "failed to lock workspace before untrusted evaluation" \
            > /logs/verifier/score_error.txt
        exit 0
    fi
    chmod a+rx /root 2>/dev/null || true
    chmod -R a+rX /root/.cache 2>/dev/null || true
fi

ORACLE_CMD_OVERRIDES_ARGS=()
if [ -r /solution/oracle_cmd_overrides.json ] \
        && [ -r /solution/oracle_cmd_overrides.token ] \
        && [ -r "${PRIVATE_META}/oracle_cmd_overrides.token" ] \
        && cmp -s /solution/oracle_cmd_overrides.token "${PRIVATE_META}/oracle_cmd_overrides.token"; then
    ORACLE_CMD_OVERRIDES_JSON="$(cat /solution/oracle_cmd_overrides.json)"
    ORACLE_CMD_OVERRIDES_ARGS=(--oracle-cmd-overrides "${ORACLE_CMD_OVERRIDES_JSON}")
fi

if [ -r /solution/oracle_env.json ] \
        && [ -r /solution/oracle_env.token ] \
        && [ -r "${PRIVATE_META}/oracle_env.token" ] \
        && cmp -s /solution/oracle_env.token "${PRIVATE_META}/oracle_env.token"; then
    while IFS= read -r env_assignment; do
        [ -n "${env_assignment}" ] || continue
        export "${env_assignment}"
    done < <("${PYTHON_BIN}" -I - <<'PY'
import json
from pathlib import Path

for key, value in json.loads(Path("/solution/oracle_env.json").read_text()).items():
    print(f"{key}={value}")
PY
    )
fi

# Provide /workspace/_task as a stable handle to sanitized runtime metadata.
# Parser, score spec, anchors, pristine state, and scorer code are absent.
if [ -n "${WORKDIR:-}" ]; then
    if ! mkdir -p "${WORKDIR}" 2>/dev/null \
            || ! rm -rf "${WORKDIR}/_task" 2>/dev/null \
            || ! ln -s "${EVAL_META}" "${WORKDIR}/_task" 2>/dev/null; then
        echo "0" > /logs/verifier/reward.txt
        echo "failed to install sanitized runtime task metadata" \
            > /logs/verifier/score_error.txt
        exit 0
    fi
fi
export TASK_DIR="${WORKDIR:-/workspace}/_task"

# Step 1: edit-range diff guard when the renderer supplied a pristine
# manifest. Older Mangrove artifacts intentionally omitted it; verification
# must still run and fail closed on eval/score errors for those artifacts.
if [ -f "${PRIVATE_META}/pristine_manifest.json" ]; then
    "${PYTHON_BIN}" -I "${PRIVATE_ROOT}/score_task.py" guard \
        --task-meta "${PRIVATE_META}" \
        --pristine "${PRIVATE_META}/pristine" \
        --workspace "${WORKDIR}" \
        --violation-out /logs/verifier/violation.txt
    guard_rc=$?

    if [ "${guard_rc}" -eq 10 ]; then
        echo "0" > /logs/verifier/reward.txt
        echo "edit-range violation — see /logs/verifier/violation.txt" >&2
        exit 0
    fi
    if [ "${guard_rc}" -ne 0 ]; then
        echo "0" > /logs/verifier/reward.txt
        echo "guard script failed unexpectedly (rc=${guard_rc})" >&2
        exit 0
    fi
else
    echo "pristine_manifest.json absent — skipping edit-range guard" >&2
fi

# Step 2: run every configured eval script, with cwd = the package root
# (config.json::files[].filename is workdir-relative; PKG_NAME is the first
# path component, e.g. "causal-learn").
#
# Heartbeat: some GPU clusters kill containers that produce no stdout for
# extended periods. Print a short line every 10 minutes so the cluster
# knows the process is still alive.
_heartbeat() {
    trap 'kill "${_SLEEP_PID:-}" 2>/dev/null || true; wait "${_SLEEP_PID:-}" 2>/dev/null || true; exit 0' HUP INT TERM
    while true; do
        sleep 600 &
        _SLEEP_PID=$!
        wait "${_SLEEP_PID}" || exit 0
        _SLEEP_PID=""
        echo "[heartbeat] $(date -u +%H:%M:%S) eval running"
    done
}
_heartbeat &
_HB_PID=$!

_stream_eval_logs() {
    trap 'kill "${_SLEEP_PID:-}" 2>/dev/null || true; wait "${_SLEEP_PID:-}" 2>/dev/null || true; exit 0' HUP INT TERM
    while true; do
        sleep "${MLSBENCH_VERIFIER_LOG_INTERVAL_SEC:-120}" &
        _SLEEP_PID=$!
        wait "${_SLEEP_PID}" || exit 0
        _SLEEP_PID=""
        echo "[verifier] live eval log tails $(date -u +%H:%M:%S)"
        _found_log=0
        for log_file in /logs/verifier/*.log; do
            [ -f "${log_file}" ] || continue
            _found_log=1
            echo "[verifier] live tail ${log_file}"
            tail -n "${MLSBENCH_VERIFIER_LIVE_TAIL_LINES:-80}" "${log_file}" || true
        done
        if [ "${_found_log}" -eq 0 ]; then
            echo "[verifier] no eval logs yet"
        fi
    done
}

_print_eval_logs() {
    echo "[verifier] eval output summary follows"
    for summary_file in \
            /logs/verifier/eval_summary.json \
            /logs/verifier/metrics.json \
            /logs/verifier/verification_result.json \
            /logs/verifier/score_error.txt \
            /logs/verifier/parse_errors.txt \
            /logs/verifier/budget_violation.txt \
            /logs/verifier/violation.txt \
            /logs/verifier/reward.txt; do
        [ -f "${summary_file}" ] || continue
        echo "[verifier] file ${summary_file}"
        sed -n '1,240p' "${summary_file}" || true
    done
    for log_file in /logs/verifier/*.log /logs/verifier/*.txt; do
        [ -f "${log_file}" ] || continue
        case "${log_file}" in
            */score_error.txt|*/parse_errors.txt|*/budget_violation.txt|*/violation.txt|*/reward.txt)
                continue
                ;;
        esac
        echo "[verifier] final tail ${log_file}"
        tail -n "${MLSBENCH_VERIFIER_FINAL_TAIL_LINES:-160}" "${log_file}" || true
    done
}

_stream_eval_logs &
_LOG_STREAM_PID=$!

_RUN_EVALS_RC=0
"${PYTHON_BIN}" -I "${PRIVATE_ROOT}/score_task.py" run-evals \
    --task-meta "${PRIVATE_META}" \
    --eval-task-meta "${EVAL_META}" \
    --workspace "${WORKDIR}" \
    --eval-root /tests/eval \
    --out-dir /logs/verifier \
    "${ORACLE_CMD_OVERRIDES_ARGS[@]}" || _RUN_EVALS_RC=$?

kill "${_HB_PID}" 2>/dev/null || true
wait "${_HB_PID}" 2>/dev/null || true
_HB_PID=""
kill "${_LOG_STREAM_PID:-}" 2>/dev/null || true
wait "${_LOG_STREAM_PID:-}" 2>/dev/null || true
_LOG_STREAM_PID=""
_print_eval_logs
if [ "${_RUN_EVALS_RC}" -ne 0 ]; then
    echo "[verifier] run-evals exited with rc=${_RUN_EVALS_RC}" >&2
    echo "0" > /logs/verifier/reward.txt
    _print_eval_logs
    exit 0
fi

# Step 3: aggregate metrics → combined_score → reward.txt.
# score_task writes only a non-authoritative dotfile beside reward.txt. The
# public reward remains the prewritten exact zero until this parent validates
# the proof and atomically renames the same-filesystem candidate. In
# particular, a verifier SIGKILL/hard timeout before that rename cannot leave
# a positive public reward; shell traps cannot run for SIGKILL.
if ! rm -f -- "${_CANDIDATE_REWARD}"; then
    printf '0\n' > /logs/verifier/reward.txt 2>/dev/null || true
    printf 'failed to clear reward candidate before scoring\n' \
        >> /logs/verifier/score_error.txt
    exit 0
fi
_SCORE_RC=0
"${PYTHON_BIN}" -I "${PRIVATE_ROOT}/score_task.py" score \
    --task-meta "${PRIVATE_META}" \
    --out-dir /logs/verifier \
    --reward-out "${_CANDIDATE_REWARD}" || _SCORE_RC=$?

_PROOF_RC=1
if [ "${_SCORE_RC}" -eq 0 ] \
        && [ -f "${_CANDIDATE_REWARD}" ] \
        && [ ! -L "${_CANDIDATE_REWARD}" ] \
        && chown 0:0 "${_CANDIDATE_REWARD}" \
        && chmod 600 "${_CANDIDATE_REWARD}"; then
    _PROOF_RC=0
    "${PYTHON_BIN}" -I - \
        /logs/verifier/verification_result.json \
        /logs/verifier/metrics.json \
        "${_CANDIDATE_REWARD}" <<'PY' || _PROOF_RC=$?
import hashlib
import json
import math
import sys
from pathlib import Path

proof_path, metrics_path, reward_path = map(Path, sys.argv[1:])
proof_text = proof_path.read_text()
proof = json.loads(proof_text)
metrics_text = metrics_path.read_text()
metrics = json.loads(metrics_text)
canonical_proof_text = json.dumps(
    proof,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=True,
    allow_nan=False,
)
canonical_metrics_text = json.dumps(
    metrics,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=True,
    allow_nan=False,
)
reward_text = reward_path.read_text()
reward = float(reward_text.strip())
if proof_text != canonical_proof_text:
    raise SystemExit(1)
if proof.get("schema_version") != 1:
    raise SystemExit(1)
if proof.get("status") != "passed" or proof.get("strict_fail_closed") is not True:
    raise SystemExit(1)
if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
    raise SystemExit(1)
if reward_text != f"{reward}\n":
    raise SystemExit(1)
if proof.get("reward") != reward or metrics.get("reward") != reward:
    raise SystemExit(1)
if metrics_text != canonical_metrics_text:
    raise SystemExit(1)
if proof.get("metrics_sha256") != hashlib.sha256(canonical_metrics_text.encode()).hexdigest():
    raise SystemExit(1)
PY
fi

if [ "${_SCORE_RC}" -ne 0 ] || [ "${_PROOF_RC}" -ne 0 ]; then
    echo "0" > /logs/verifier/reward.txt
    _remove_reward_candidate
    rm -f /logs/verifier/verification_result.json /logs/verifier/metrics.json
    printf 'strict score stage failed or produced invalid success proof (score_rc=%s proof_rc=%s)\n' \
        "${_SCORE_RC}" "${_PROOF_RC}" \
        >> /logs/verifier/score_error.txt
else
    # This same-directory rename is the only positive reward publication. A
    # kill before it leaves zero; a kill after it is safe because proof passed.
    if mv -f -- "${_CANDIDATE_REWARD}" /logs/verifier/reward.txt; then
        _VERIFICATION_COMMITTED=1
    else
        _VERIFICATION_COMMITTED=0
        _remove_reward_candidate
        printf '0\n' > /logs/verifier/reward.txt 2>/dev/null || true
        rm -f /logs/verifier/verification_result.json /logs/verifier/metrics.json
        printf 'failed to atomically publish verified reward\n' \
            >> /logs/verifier/score_error.txt
    fi
fi

_print_eval_logs

exit 0
