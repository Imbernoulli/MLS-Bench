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
#   /tests/eval/scripts/*.sh       (every eval script, visible + hidden)

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

mkdir -p /logs/verifier
# Pre-write reward before any auxiliary diagnostics so interruption always
# leaves an explicit exact zero for the platform to collect.
printf '0\n' > /logs/verifier/reward.txt
# Interpreter hashing is diagnostic only. Bound it so slow node/layer I/O
# cannot block the actual training and evaluation pipeline.
if command -v timeout >/dev/null 2>&1; then
    timeout --kill-after=5s 30s "${PYTHON_BIN}" -I -c \
        "import hashlib,sys; print('verifier_python_sha256='+hashlib.sha256(open(sys.executable,'rb').read()).hexdigest())" \
        > /logs/verifier/python_audit.txt 2>&1 \
        || printf 'verifier python audit unavailable or timed out\n' \
            > /logs/verifier/python_audit.txt
else
    printf 'verifier python audit skipped: timeout command unavailable\n' \
        > /logs/verifier/python_audit.txt
fi
_CANDIDATE_REWARD="/logs/verifier/.reward.candidate"
rm -f -- "${_CANDIDATE_REWARD}"
_VERIFICATION_COMMITTED=0

TASK_ID="$(cat /tests/meta/task_id 2>/dev/null || echo unknown)"
PKG_NAME="$(cat /tests/meta/package 2>/dev/null || echo unknown)"
WORKDIR="$(cat /tests/meta/workdir 2>/dev/null || echo /workspace)"

PRIVATE_ROOT="$(mktemp -d /tmp/mlsbench-verifier.XXXXXX)"
PRIVATE_META="${PRIVATE_ROOT}/meta"
cp -a /tests/meta "${PRIVATE_META}"
cp /tests/score_task.py "${PRIVATE_ROOT}/score_task.py"
chmod -R a-w /tests/meta
chmod -R a-w "${PRIVATE_META}" "${PRIVATE_ROOT}/score_task.py"
chmod go-rwx "${PRIVATE_ROOT}" || true

_cleanup_verifier() {
    if [ "${_VERIFICATION_COMMITTED:-0}" -ne 1 ]; then
        printf '0\n' > /logs/verifier/reward.txt 2>/dev/null || true
    fi
    rm -f -- "${_CANDIDATE_REWARD}" 2>/dev/null || true
    if [ -n "${_HB_PID:-}" ]; then
        kill "${_HB_PID}" 2>/dev/null || true
        wait "${_HB_PID}" 2>/dev/null || true
        _HB_PID=""
    fi
    chmod -R u+w "${PRIVATE_ROOT}" 2>/dev/null || true
    rm -rf "${PRIVATE_ROOT}"
}
trap _cleanup_verifier EXIT
trap 'exit 0' HUP INT TERM

# Provide /workspace/_task as a stable handle to the verifier-only meta dir.
# Several tasks' eval scripts and edit_ops resolve task files (data/, task_description.md)
# through TASK_DIR or /workspace/_task, expecting this to exist at eval time.
if [ -n "${WORKDIR:-}" ]; then
    mkdir -p "${WORKDIR}" 2>/dev/null || true
    rm -rf "${WORKDIR}/_task" 2>/dev/null || true
    ln -s "${PRIVATE_META}" "${WORKDIR}/_task" 2>/dev/null || true
fi
export TASK_DIR="${WORKDIR:-/workspace}/_task"

# Step 1: edit-range diff guard. The pristine baseline is the
# per-task-rendered tree under tests/meta/pristine/ (mounted only at verify
# time), so the agent had no opportunity to tamper with it.
# Skip guard when pristine_manifest.json is absent (Mangrove mode without
# vendor source — the guard cannot run without the pristine baseline).
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

# Step 2: run every eval script (visible + hidden), with cwd = the package root
# (config.json::files[].filename is workdir-relative; PKG_NAME is the first
# path component, e.g. "causal-learn").
#
# Heartbeat: some GPU clusters kill containers that produce no stdout for
# extended periods. Print a short line every 10 minutes so the cluster
# knows the process is still alive.
_heartbeat() {
    while sleep 600; do
        echo "[heartbeat] $(date -u +%H:%M:%S) eval running"
    done
}
if [ "${MLSBENCH_DISABLE_HEARTBEAT:-0}" = 1 ]; then
    _HB_PID=""
else
    _heartbeat &
    _HB_PID=$!
fi

_RUN_EVALS_RC=0
"${PYTHON_BIN}" -I "${PRIVATE_ROOT}/score_task.py" run-evals \
    --task-meta "${PRIVATE_META}" \
    --workspace "${WORKDIR}" \
    --eval-root /tests/eval \
    --out-dir /logs/verifier || _RUN_EVALS_RC=$?

kill "${_HB_PID}" 2>/dev/null || true
wait "${_HB_PID}" 2>/dev/null || true
_HB_PID=""

if [ "${_RUN_EVALS_RC}" -ne 0 ]; then
    printf '0\n' > /logs/verifier/reward.txt
    exit 0
fi

# Step 3: aggregate metrics → combined_score → reward.txt.
_SCORE_RC=0
"${PYTHON_BIN}" -I "${PRIVATE_ROOT}/score_task.py" score \
    --task-meta "${PRIVATE_META}" \
    --out-dir /logs/verifier \
    --reward-out "${_CANDIDATE_REWARD}" || _SCORE_RC=$?

_PROOF_RC=1
if [ "${_SCORE_RC}" -eq 0 ] && [ -f "${_CANDIDATE_REWARD}" ]; then
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
reward_text = reward_path.read_text()
reward = float(reward_text.strip())
canonical_proof = json.dumps(
    proof,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=True,
    allow_nan=False,
)
canonical_metrics = json.dumps(
    metrics,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=True,
    allow_nan=False,
)
if proof_text != canonical_proof or proof.get("schema_version") != 1:
    raise SystemExit(1)
if proof.get("status") != "passed" or proof.get("strict_fail_closed") is not True:
    raise SystemExit(1)
if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
    raise SystemExit(1)
if reward_text != f"{reward}\n":
    raise SystemExit(1)
if proof.get("reward") != reward or metrics.get("reward") != reward:
    raise SystemExit(1)
if metrics_text != canonical_metrics:
    raise SystemExit(1)
if proof.get("metrics_sha256") != hashlib.sha256(metrics_text.encode()).hexdigest():
    raise SystemExit(1)
PY
fi

if [ "${_SCORE_RC}" -eq 0 ] && [ "${_PROOF_RC}" -eq 0 ] \
        && mv -f -- "${_CANDIDATE_REWARD}" /logs/verifier/reward.txt; then
    _VERIFICATION_COMMITTED=1
else
    printf '0\n' > /logs/verifier/reward.txt
    rm -f /logs/verifier/verification_result.json /logs/verifier/metrics.json
    printf 'score stage failed or produced invalid success proof (score_rc=%s proof_rc=%s)\n' \
        "${_SCORE_RC}" "${_PROOF_RC}" >> /logs/verifier/score_error.txt
fi

exit 0
