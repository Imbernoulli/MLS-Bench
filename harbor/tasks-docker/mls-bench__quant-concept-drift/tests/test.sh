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
#   /tests/eval/scripts/*.sh       (every setting's eval script)

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
# Root-side python is later invoked with -S (not -I) plus a trusted PYTHONPATH
# — see the "Verifier interpreter hardening" block below for why -S is needed
# (it is the only flag that stops sitecustomize / usercustomize / *.pth code
# from running inside the root verifier) and how numpy/torch stay importable.
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

# --------------------------------------------------------------------------- #
# Verifier interpreter hardening.
#
# The agent phase runs as root in this same container, so it can plant code
# that executes at interpreter STARTUP inside the (root) verifier process and
# thereby defeats the uid-65534 eval drop entirely — reading /tests/meta,
# unsealing the secret dirs, or rewriting reward.txt. Measured vectors (see
# docs/harbor.md "A root agent-phase can execute code inside the verifier"):
#
#   * site-packages/sitecustomize.py  — runs under -I (only -S blocks it)
#   * a *.pth file whose line starts `import ...` — same, runs under -I
#   * site-packages/usercustomize.py  — blocked by -I's implied -s, still runs
#                                        under a bare interpreter
#   * replacing /usr/local/bin/python3 (a root-writable shim/symlink) — runs
#                                        whatever flags we pass
#
# Two defences, both here so a re-render / verifier drop-in propagates them to
# every task with no per-task change:
#
#   1. Refuse a non-ELF interpreter. A real CPython is an ELF binary; the
#      demonstrated shim was a 42-byte `#!` script. This kills the cheap
#      interpreter-replacement vector. If tests/meta/interpreter_manifest.json
#      is present (baked post-build, see scripts/harbor/bake_interpreter_manifest.py)
#      we additionally require the interpreter's sha256 to match it and refuse
#      on drift — the strong form, against a compiled ELF replacement.
#   2. Run every ROOT-side python (this snapshot + all three score_task.py
#      sub-commands) with `-S`, which stops site.py from importing
#      sitecustomize / usercustomize and from executing *.pth code, while
#      `PYTHONPATH=<genuine site dir(s)>` (computed under -S -E, so no hook can
#      run during the computation itself) keeps numpy / torch / mlsbench
#      importable. `-S` is used INSTEAD of `-I` because `-I` ignores PYTHONPATH;
#      test.sh already scrubbed the agent's PYTHON* env above and a script (not
#      -c/-m) is run, so -I buys nothing over -S here. The eval CHILDREN keep
#      their own flags: they already run as uid 65534, so startup code there is
#      contained by the drop, not an escalation.
# --------------------------------------------------------------------------- #
refuse_verify() {
    # $1: human message. Fail closed: reward 0, exit 0 (a graded non-pass).
    echo "0" > /logs/verifier/reward.txt
    echo "verifier hardening: $1" >&2
    echo "$1" >> /logs/verifier/python_audit.txt 2>/dev/null || true
    exit 0
}

# Resolve what ACTUALLY executes, not the path we were handed. Several images
# ship /usr/local/bin/python3 as a `#!/bin/sh` wrapper around a venv, on purpose,
# because Harbor resets PATH and would otherwise lose /opt/venv/bin. Checking the
# wrapper's magic rejects those images with "not an ELF binary (magic=23212f62)"
# — 23212f62 is `#!/b` — on every venue, for a legitimate layout. sys.executable
# (computed with hooks disabled) is the real interpreter in both layouts, and it
# is also what the sha256 manifest below is taken over.
REAL_PY="$("${PYTHON_BIN}" -S -E -c 'import sys;print(sys.executable)' 2>/dev/null || true)"
[ -n "${REAL_PY}" ] || REAL_PY="$(readlink -f "${PYTHON_BIN}" 2>/dev/null || echo "${PYTHON_BIN}")"
REAL_PY="$(readlink -f "${REAL_PY}" 2>/dev/null || echo "${REAL_PY}")"
PY_SHA="$("${PYTHON_BIN}" -S -E -c "import hashlib,sys;print(hashlib.sha256(open(sys.executable,'rb').read()).hexdigest())" 2>/dev/null || echo unknown)"
echo "verifier_python=${REAL_PY}" >> /logs/verifier/python_audit.txt
echo "verifier_python_sha256=${PY_SHA}" >> /logs/verifier/python_audit.txt

# 1a. non-ELF interpreter = a shim the agent dropped in place of python.
PY_MAGIC="$(head -c 4 "${REAL_PY}" 2>/dev/null | od -An -tx1 2>/dev/null | tr -d ' \n')"
if [ "${PY_MAGIC}" != "7f454c46" ]; then
    refuse_verify "verifier interpreter ${REAL_PY} is not an ELF binary (magic=${PY_MAGIC:-none}) — likely an agent-planted shim"
fi
# 1b. optional strong form: sha256 must match the build-time manifest.
if [ -r /tests/meta/interpreter_manifest.json ]; then
    EXPECTED_SHA="$("${PYTHON_BIN}" -S -E -c "import json,sys;print(json.load(open('/tests/meta/interpreter_manifest.json')).get('sha256',''))" 2>/dev/null || echo "")"
    if [ -n "${EXPECTED_SHA}" ] && [ "${EXPECTED_SHA}" != "${PY_SHA}" ]; then
        refuse_verify "verifier interpreter sha256 ${PY_SHA} != manifest ${EXPECTED_SHA} — interpreter tampered after build"
    fi
fi

# 1c. optional: the interpreters' whole import trees (stdlib, site-packages,
#     sitecustomize/.pth hooks) must match the build-time site manifest
#     (tests/meta/site_manifest.json, baked post-build by
#     scripts/harbor/bake_interpreter_manifest.py). Files added after build are
#     removed, planted bytecode is removed, a modified or deleted file refuses
#     the run. This is what protects the TRUSTED root halves of keep-root tasks,
#     which run a plain python3 that loads `site`, and every root process that
#     imports numpy/torch. Runs before the guard and before any eval command.
if [ -r /tests/meta/site_manifest.json ]; then
    "${PYTHON_BIN}" -S -E -B /tests/score_task.py site-check \
        --manifest /tests/meta/site_manifest.json --out /logs/verifier/site_check.txt
    site_rc=$?
    if [ "${site_rc}" -ne 0 ]; then
        refuse_verify "site-check rc=${site_rc}: the build-time Python environment was modified after build (see site_check.txt)"
    fi
elif [ "${MLSB_VERIFIER_MODE:-}" = "separate" ] \
        && tr '\0' '\n' < /proc/1/environ 2>/dev/null | grep -qx 'MLSB_VERIFIER_MODE=separate'; then
    # 1d'. separate-verifier mode (Harbor >= 0.13): this container was built
    #      fresh from tests/Dockerfile, which alone sets MLSB_VERIFIER_MODE; the
    #      agent never ran here, so there is nothing to check. Both the env and
    #      pid 1's env (from the image config, out of a shared-mode agent's
    #      reach) must carry it. Missing it only runs the check on a pristine
    #      image, which tests/meta/record_check_allow.json makes pass.
    echo "record-check: skipped (separate verifier container, pristine image)" \
        > /logs/verifier/site_check.txt
else
    # 1d. no baked manifest (the shipped bundles' case): fall back to the
    #     BUILD-FREE check, which uses the pip RECORD / conda-meta hashes that
    #     already ship in the image. It refuses on a modified recorded
    #     dependency file (numpy/torch/the research package/a recorded .so) and
    #     removes planted sitecustomize / usercustomize / *.pth / new top-level
    #     modules. It is strictly weaker than the baked manifest (it cannot see
    #     the stdlib, which pip/conda do not record, or files neither installer
    #     records), but it needs no per-image bake, so it runs on every bundle.
    #     Changes the image BUILD itself makes (a Dockerfile step patching an
    #     installed file, a setup.py-installed top-level module) are accepted
    #     only at the hashes measured on a clean build (tests/meta allowlist).
    "${PYTHON_BIN}" -S -E -B /tests/score_task.py record-check \
        --mode enforce --allow /tests/meta/record_check_allow.json \
        --out /logs/verifier/site_check.txt
    rc_rc=$?
    if [ "${rc_rc}" -eq 10 ]; then
        refuse_verify "record-check: a recorded dependency file was modified after build (see site_check.txt)"
    elif [ "${rc_rc}" -ne 0 ]; then
        echo "record-check rc=${rc_rc} (non-fatal); import-tree integrity only partially checked" \
            >> /logs/verifier/site_check.txt
    fi
fi

# 2. genuine system site-packages, computed with hooks disabled (-S -E).
VERIFIER_SITE="$("${PYTHON_BIN}" -S -E -c '
import sysconfig
paths = []
for p in (sysconfig.get_path("purelib"), sysconfig.get_path("platlib")):
    if p and p not in paths:
        paths.append(p)
try:
    import site
    for p in site.getsitepackages():   # system site dirs, NOT the user site
        if p and p not in paths:
            paths.append(p)
except Exception:
    pass
print(":".join(paths))
' 2>/dev/null)"
# 2b. editable installs. -S also skips *.pth processing, which is how
# `pip install -e` makes a package importable; a verifier module that imports
# such a package (e.g. causallearn, installed editable from /workspace) then
# fails. Recover the locations WITHOUT executing anything -- a .pth line that
# names a directory is taken, a line starting with `import` is code and is
# never run, and a PEP 660 `__editable___*_finder.py` is read with
# ast.literal_eval for its MAPPING. These locations are handed ONLY to the
# `score` step (which runs after the guard has verified the workspace) and are
# APPENDED to sys.path there, so neither the stdlib, the site dirs nor the
# staged harness can be shadowed by them; the guard never sees them.
VERIFIER_EDITABLE="$(VERIFIER_SITE="${VERIFIER_SITE}" "${PYTHON_BIN}" -S -E -c '
import ast, glob, os
paths = [p for p in os.environ.get("VERIFIER_SITE", "").split(":") if p]
extra = []
for sp in paths:
    for pth in sorted(glob.glob(os.path.join(sp, "*.pth"))):
        try:
            lines = open(pth, encoding="utf-8").read().splitlines()
        except Exception:
            continue
        for ln in lines:
            s = ln.strip()
            if not s or s.startswith("#") or s.split()[0] == "import":
                continue
            d = s if os.path.isabs(s) else os.path.join(sp, s)
            if os.path.isdir(d) and d not in paths and d not in extra:
                extra.append(d)
    for fin in sorted(glob.glob(os.path.join(sp, "__editable___*_finder.py"))):
        try:
            tree = ast.parse(open(fin, encoding="utf-8").read())
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "MAPPING" for t in node.targets):
                try:
                    mapping = ast.literal_eval(node.value)
                except Exception:
                    continue
                for pkg_path in (mapping.values() if isinstance(mapping, dict) else []):
                    d = os.path.dirname(str(pkg_path))
                    if os.path.isdir(d) and d not in paths and d not in extra:
                        extra.append(d)
print(":".join(extra))
' 2>/dev/null)"
# Root-side python is invoked as `PYTHONPATH="${VERIFIER_SITE}" "${PYTHON_BIN}" -S`
# (see each score_task.py call below). PYTHONPATH is set inline, never exported,
# so the trusted site path is not forced into the uid-65534 eval children's
# environment (they resolve their own site normally / via their package env).

set -uo pipefail

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

# score_task.py seals /tests/meta and /solution (chmod 0700) while the eval
# commands run, because those run the SUBMISSION's code, and file modes alone
# do not contain a root process -- `chmod -R a-w` above does not stop root. It restores them
# itself; this trap is the belt-and-braces for a verifier that is killed, since
# both are bind mounts of the bundle on the host and leaving them 0700 would
# break the next `harbor run` for a non-root invoking user.
TESTS_META_MODE="$(stat -c %a /tests/meta 2>/dev/null || echo 755)"
SOLUTION_MODE="$(stat -c %a /solution 2>/dev/null || echo 755)"
SCORING_ASSETS=/tests/mlsbench_src/mlsbench/scoring/assets
SCORING_ASSETS_MODE="$(stat -c %a "$SCORING_ASSETS" 2>/dev/null || echo 755)"
restore_bundle_modes() {
    chmod "${TESTS_META_MODE}" /tests/meta 2>/dev/null || true
    [ -d /solution ] && chmod "${SOLUTION_MODE}" /solution 2>/dev/null || true
    [ -d "$SCORING_ASSETS" ] && chmod "$SCORING_ASSETS_MODE" "$SCORING_ASSETS" 2>/dev/null || true
}
trap 'rm -rf "${PRIVATE_ROOT}"; restore_bundle_modes' EXIT

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
PYTHONPATH="${VERIFIER_SITE}" "${PYTHON_BIN}" -S -B "${PRIVATE_ROOT}/score_task.py" guard \
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

# Step 2: run every setting's eval script, with cwd = the package root
# (config.json::files[].filename is workdir-relative; PKG_NAME is the first
# path component, e.g. "causal-learn").
PYTHONPATH="${VERIFIER_SITE}" "${PYTHON_BIN}" -S -B "${PRIVATE_ROOT}/score_task.py" run-evals \
    --task-meta "${PRIVATE_META}" \
    --workspace "${WORKDIR}" \
    --eval-root /tests/eval \
    --out-dir /logs/verifier \
    "${ORACLE_CMD_OVERRIDES_ARGS[@]}"

# Step 3: aggregate metrics → combined_score → reward.txt.
MLSBENCH_VERIFIER_EXTRA_PATH="${VERIFIER_EDITABLE}" PYTHONPATH="${VERIFIER_SITE}" "${PYTHON_BIN}" -S -B "${PRIVATE_ROOT}/score_task.py" score \
    --task-meta "${PRIVATE_META}" \
    --out-dir /logs/verifier \
    --reward-out /logs/verifier/reward.txt

# Step 4: reclaim SAVE_PATH. score_task.py puts SAVE_PATH/OUTPUT_DIR under
# /logs/verifier (`save/<task>/harbor/seed_N`), which for a fine-tuning task
# means the whole trained checkpoint -- and Harbor downloads the verifier log
# directory WHOLE by tarring it on the machine and pulling the archive back
# (harbor/environments/tar_transfer.py, through a single 600 s exec). Weights
# are not logs: nothing after this point reads them, since reward.txt,
# metrics.json, eval_summary.json and the per-setting logs all sit directly in
# /logs/verifier. Leaving them there makes the verifier-dir archive too large
# to download (`DownloadVerifierDirError`), which loses the whole trial after
# the oracle, the guard and every eval have run cleanly.
# The sizes are recorded first so the reclamation is visible in the artefacts.
if [ -d /logs/verifier/save ]; then
    # Two invocations, not `du -sh A B`: GNU du counts each file once across
    # its arguments, so naming the parent first makes the nested one print
    # nothing at all and the artefact loses the number worth having.
    du -sh /logs/verifier/save > /logs/verifier/save_reclaimed.txt 2>&1 || true
    du -sh /logs/verifier >> /logs/verifier/save_reclaimed.txt 2>&1 || true
    rm -rf /logs/verifier/save 2>/dev/null \
        || echo "could not fully remove /logs/verifier/save" >> /logs/verifier/save_reclaimed.txt
fi

exit 0
