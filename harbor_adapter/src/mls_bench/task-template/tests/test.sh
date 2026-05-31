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

# Snapshot the verifier python's integrity so a tampered binary is at least
# logged for postmortem. We cannot prevent a root agent from replacing it,
# but a mismatch against the base image's recorded sha lets us flag it.
"${PYTHON_BIN}" -c "import hashlib,sys; print('verifier_python_sha256='+hashlib.sha256(open(sys.executable,'rb').read()).hexdigest())" 2>&1 \
    | tee -a /logs/verifier/python_audit.txt >/dev/null || true

set -uo pipefail

mkdir -p /logs/verifier
# Pre-write reward so Mangrove always finds a file even if we get killed.
echo "0" > /logs/verifier/reward.txt

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
trap 'rm -rf "${PRIVATE_ROOT}"' EXIT

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
_heartbeat() { while sleep 600; do echo "[heartbeat] $(date -u +%H:%M:%S) eval running"; done; }
_heartbeat &
_HB_PID=$!

"${PYTHON_BIN}" -I "${PRIVATE_ROOT}/score_task.py" run-evals \
    --task-meta "${PRIVATE_META}" \
    --workspace "${WORKDIR}" \
    --eval-root /tests/eval \
    --out-dir /logs/verifier

kill $_HB_PID 2>/dev/null || true

# Step 3: aggregate metrics → combined_score → reward.txt.
"${PYTHON_BIN}" -I "${PRIVATE_ROOT}/score_task.py" score \
    --task-meta "${PRIVATE_META}" \
    --out-dir /logs/verifier \
    --reward-out /logs/verifier/reward.txt

exit 0
