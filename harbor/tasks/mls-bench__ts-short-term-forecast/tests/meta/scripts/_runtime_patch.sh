#!/bin/bash
# Runtime patch sourced by every ts-short-term-forecast eval/baseline script.
#
# All three m4_* labels (and each baseline / the oracle override, and every
# seed) run concurrently in ONE workspace, and every run writes to the shared
# leaf ./m4_results/<model>/. The pristine Time-Series-Library code creates it
# with a check-then-create TOCTOU:
#     if not os.path.exists(folder_path):
#         os.makedirs(folder_path)
# The loser of the race dies with FileExistsError AFTER training completes,
# losing its metric. Fresh workspaces get the fix from
# vendor/pkg_configs/Time-Series-Library/pre_edit.py, but workspaces baked
# before that fix (reused native workspaces, the Harbor image's baked
# /workspace) still carry the bug — so patch the exact old block to
# os.makedirs(..., exist_ok=True) here before launching run.py. Anchored on
# the old block, idempotent, and a no-op once the workspace already has the
# fix. Eval scripts run with cwd = the package root, so the path is relative.

_tsl_patch_lock="${TSL_RUNTIME_PATCH_LOCK:-.tsl_runtime_patch.lock}"
{
if command -v flock >/dev/null 2>&1; then
    flock 9
fi

python3 - <<'PY'
from pathlib import Path

exp_py = Path("exp/exp_short_term_forecasting.py")
if exp_py.exists():
    text = exp_py.read_text()
    old = (
        "        folder_path = './m4_results/' + self.args.model + '/'\n"
        "        if not os.path.exists(folder_path):\n"
        "            os.makedirs(folder_path)\n"
    )
    new = (
        "        folder_path = './m4_results/' + self.args.model + '/'\n"
        "        os.makedirs(folder_path, exist_ok=True)\n"
    )
    if old in text:
        exp_py.write_text(text.replace(old, new, 1))
PY
} 9>"${_tsl_patch_lock}"
