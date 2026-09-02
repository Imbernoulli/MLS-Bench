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
# We still pass the isolation flags below (-I -B, plus -P where the
# interpreter is new enough) on every invocation so the agent's planted
# PYTHON* envs, user site, script directory and current directory are not on
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

mkdir -p /logs/verifier

# Isolation flags for every verifier-side interpreter:
#
#   -I  isolated mode. Implies -E and -s, and (since 3.4) drops the script's
#       own directory from sys.path, so a module planted next to the script
#       cannot shadow a stdlib import.
#   -B  never write bytecode, so a verifier process cannot leave a .pyc
#       behind that a later phase would load.
#   -P  (3.11+) the explicit "keep sys.path[0] out" switch. Older
#       interpreters do not merely ignore it -- python:3.8 / python:3.10 base
#       images abort with "Unknown option: -P" -- and several task images are
#       on those, so probe for it rather than assume. On those versions -I
#       already provides what -P would add.
PY_SAFE_FLAGS=(-I -B)
if "${PYTHON_BIN}" -P -c "" >/dev/null 2>&1; then
    PY_SAFE_FLAGS+=(-P)
fi
export MLSBENCH_VERIFIER_PY_SAFE_FLAGS="${PY_SAFE_FLAGS[*]}"
# Nothing the verifier starts -- including the eval scripts -- should write
# bytecode into the workspace.
export PYTHONDONTWRITEBYTECODE=1

# Snapshot the verifier python's integrity so a tampered binary is at least
# logged for postmortem. We cannot prevent a root agent from replacing it,
# but a mismatch against the base image's recorded sha lets us flag it.
"${PYTHON_BIN}" "${PY_SAFE_FLAGS[@]}" -c "import hashlib,sys; print('verifier_python_sha256='+hashlib.sha256(open(sys.executable,'rb').read()).hexdigest())" 2>&1 \
    | tee -a /logs/verifier/python_audit.txt >/dev/null || true

set -uo pipefail

TASK_ID="$(cat /tests/meta/task_id 2>/dev/null || echo unknown)"
PKG_NAME="$(cat /tests/meta/package 2>/dev/null || echo unknown)"
WORKDIR="$(cat /tests/meta/workdir 2>/dev/null || echo /workspace)"

# Purge compiled Python from the workspace before ANY code runs.
#
# `__pycache__/` and `*.pyc` are deliberately excluded both from
# tests/meta/pristine_manifest.json and from the verifier's workspace walk --
# a manifest entry the walk skips is reported as a deleted file, which used to
# reject every submission. The price of that exclusion is that bytecode is
# invisible to the integrity guard: a submission can leave a sourceless
# `<pkg>/json.pyc`, or a `__pycache__` entry whose header forges the source's
# mtime and size, and Python will execute it in place of the read-only module
# it shadows. Eval scripts start their evaluator as `python <pkg>/<eval>.py`,
# which puts the agent-writable package dir on sys.path[0], so this is a live
# path to a forged metric -- reproduced end to end on more than one package.
#
# Deleting it is safe and cheap: bytecode with a source beside it is
# regenerated on import, and bytecode with no source is exactly what we are
# refusing to run.
purge_bytecode() {
    local root="$1"
    [ -n "${root}" ] && [ -d "${root}" ] || return 0
    find "${root}" -type d -name '__pycache__' -prune -print -exec rm -rf {} + 2>/dev/null
    find "${root}" -type f \( -name '*.pyc' -o -name '*.pyo' \) -print -delete 2>/dev/null
    return 0
}
{
    echo "=== bytecode purge before verification ==="
    purge_bytecode "${WORKDIR}"
    [ "${WORKDIR}" = "/workspace" ] || purge_bytecode /workspace
} >>/logs/verifier/bytecode_purge.txt 2>&1

PRIVATE_ROOT="$(mktemp -d /tmp/mlsbench-verifier.XXXXXX)"
PRIVATE_META="${PRIVATE_ROOT}/meta"
cp -a /tests/meta "${PRIVATE_META}"
cp /tests/score_task.py "${PRIVATE_ROOT}/score_task.py"
chmod -R a-w /tests/meta
chmod -R a-w "${PRIVATE_META}" "${PRIVATE_ROOT}/score_task.py"
chmod go-rwx "${PRIVATE_ROOT}" || true
trap 'rm -rf "${PRIVATE_ROOT}"' EXIT

ORACLE_CMD_OVERRIDES_ARGS=()
if [ -r /solution/oracle_cmd_overrides.json ] \
        && [ -r /solution/oracle_cmd_overrides.token ] \
        && [ -r "${PRIVATE_META}/oracle_cmd_overrides.token" ] \
        && cmp -s /solution/oracle_cmd_overrides.token "${PRIVATE_META}/oracle_cmd_overrides.token"; then
    ORACLE_CMD_OVERRIDES_JSON="$(cat /solution/oracle_cmd_overrides.json)"
    ORACLE_CMD_OVERRIDES_ARGS=(--oracle-cmd-overrides "${ORACLE_CMD_OVERRIDES_JSON}")
fi

# Step 1: edit-range diff guard. The pristine baseline is the
# per-task-rendered tree under tests/meta/pristine/ (mounted only at verify
# time), so the agent had no opportunity to tamper with it.
"${PYTHON_BIN}" "${PY_SAFE_FLAGS[@]}" "${PRIVATE_ROOT}/score_task.py" guard \
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

# Step 2: run every eval script (visible + hidden), with cwd = the package root
# (config.json::files[].filename is workdir-relative; PKG_NAME is the first
# path component, e.g. "causal-learn").
"${PYTHON_BIN}" "${PY_SAFE_FLAGS[@]}" "${PRIVATE_ROOT}/score_task.py" run-evals \
    --task-meta "${PRIVATE_META}" \
    --workspace "${WORKDIR}" \
    --eval-root /tests/eval \
    --out-dir /logs/verifier \
    "${ORACLE_CMD_OVERRIDES_ARGS[@]}"

# Step 3: aggregate metrics → combined_score → reward.txt.
"${PYTHON_BIN}" "${PY_SAFE_FLAGS[@]}" "${PRIVATE_ROOT}/score_task.py" score \
    --task-meta "${PRIVATE_META}" \
    --out-dir /logs/verifier \
    --reward-out /logs/verifier/reward.txt

exit 0
