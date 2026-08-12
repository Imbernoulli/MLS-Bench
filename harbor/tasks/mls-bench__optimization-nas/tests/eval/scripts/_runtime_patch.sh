#!/bin/bash
# Verifier-only runtime patch for optimization-nas.
#
# The agent workspace image bakes naslib/custom_nas_search.py from
# environment/_scaffold at image-build time. A stale image may still carry
# the pre-wrapper FIXED entry, whose _load_val_table only reads the staged
# table from disk (no _PRELOADED_INPUTS protocol) — the fixed wrapper
# fixed_entry.py would then fall back to the legacy direct-launch flow.
# Splice in the current fixed blocks before the eval runs. Every replacement
# is an anchored exact-string match against FIXED-region text only (the
# agent's editable region, lines 163-234, is never touched): it applies only
# when the stale block is present and is a no-op on an up-to-date workspace.

_nas_patch_lock="${NAS_RUNTIME_PATCH_LOCK:-/workspace/.nas_runtime_patch.lock}"
{
if command -v flock >/dev/null 2>&1; then
    flock 9
fi

python3 - <<'PY'
import os
import tempfile

TARGET = "/workspace/naslib/custom_nas_search.py"

REPLACEMENTS = [
    # round-7: fixed-entry wrapper protocol (the wrapper preloads + unlinks
    # the staged table BEFORE this module is imported; _load_val_table
    # consults _PRELOADED_INPUTS and falls back to disk on direct launches).
    (
        '''# =====================================================================
# FIXED: Main entry point — search + final-architecture report
# =====================================================================
def _load_val_table(env_name, seed):
    """Load this run's validation table and (when the harness marks the
    materialized inputs as ephemeral, i.e. re-created for every evaluation)
    delete the on-disk copy BEFORE any editable code runs."""
    path = (
        Path(__file__).resolve().parent
        / "naslib"
        / "data"
        / f"nb201_tables_{env_name}_s{seed}.json"
    )
    if not path.exists():
        print(f"ERROR: validation table not found: {path}", flush=True)
        sys.exit(1)
    with open(path) as f:
        tables = json.load(f)
    if os.environ.get("MLSBENCH_EPHEMERAL_INPUTS") == "1":
        try:
            os.remove(path)
        except OSError:
            pass
    return tables["val"]
''',
        '''# =====================================================================
# FIXED: Main entry point — search + final-architecture report
# =====================================================================
# Set by the FIXED wrapper (scripts/fixed_entry.py) AFTER it read and
# unlinked the staged validation table and BEFORE this module was imported:
# maps blob basename -> file content. None when the module is launched
# directly — _load_val_table then reads the on-disk table (legacy flow).
_PRELOADED_INPUTS: dict[str, str] | None = None


def _load_val_table(env_name, seed):
    """Load this run's validation table. Prefers the payload preloaded by the
    FIXED wrapper (which read and unlinked the table before any editable code
    could run); falls back to the on-disk copy when launched directly, deleting
    it after loading when the harness marks the materialized inputs as
    ephemeral (i.e. re-created for every evaluation)."""
    global _PRELOADED_INPUTS
    name = f"nb201_tables_{env_name}_s{seed}.json"
    preloaded = _PRELOADED_INPUTS
    # Drop the raw payloads before any editable code runs: from here on the
    # table lives only in this fixed entry's scope (handed to BenchmarkAPI).
    _PRELOADED_INPUTS = None
    payload = (preloaded or {}).pop(name, None)
    if payload is not None:
        tables = json.loads(payload)
        return tables["val"]
    path = Path(__file__).resolve().parent / "naslib" / "data" / name
    if not path.exists():
        print(f"ERROR: validation table not found: {path}", flush=True)
        sys.exit(1)
    with open(path) as f:
        tables = json.load(f)
    if os.environ.get("MLSBENCH_EPHEMERAL_INPUTS") == "1":
        try:
            os.remove(path)
        except OSError:
            pass
    return tables["val"]
''',
    ),
]

if os.path.exists(TARGET):
    with open(TARGET, "r", encoding="utf-8") as fh:
        text = fh.read()
    patched = text
    for old, new in REPLACEMENTS:
        if old in patched:
            patched = patched.replace(old, new, 1)
    if patched != text:
        target_dir = os.path.dirname(TARGET)
        fd, tmp = tempfile.mkstemp(dir=target_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(patched)
            os.replace(tmp, TARGET)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        print(f"[runtime-patch] updated {TARGET}", flush=True)
PY
} 9>"${_nas_patch_lock}"
