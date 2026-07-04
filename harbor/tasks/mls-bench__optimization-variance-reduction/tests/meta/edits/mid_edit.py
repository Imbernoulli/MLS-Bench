"""Mid-edit: create the opt-variance-reduction scaffold.

Creates the agent-editable optimizer module ``custom_vr.py`` and the FIXED
``vr_driver.py`` driver, AND pre-generates the observable arrays for the
synthetic ``conditioned`` problem.

The dataset loaders — including the synthetic ground-truth ``w_true`` — live
ONLY in ``holdout/optimization-variance-reduction/dgp.py`` (host-side, never
bind-mounted). Here we import it host-side and write ONLY the observable
(X_train, y_train, X_test, y_test) for ``conditioned`` into the workspace, so the
synthetic generator and w_true never enter the container. Inputs are
byte-identical to the originals, so honest results are unchanged.

Public-image datasets (MNIST/CIFAR-10) are NOT pre-generated (they are ~0.3/1.0
GB each); ``vr_driver.py`` reads them from the baked /data dir at run time. The
seal for those problems is that the loaders, ``evaluate``, and the test labels
are no longer in the agent-editable module's scope.
"""

import base64
import io
import json
import sys
from pathlib import Path

import numpy as np

_DIR = Path(__file__).parent
_TASK_DIR = _DIR.parent
_PROJECT_ROOT = _TASK_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "holdout" / "optimization-variance-reduction"))
import dgp  # host-only

_CUSTOM_PY = (_DIR / "custom_template.py").read_text()
_DRIVER_PY = (_DIR / "vr_driver_template.py").read_text()

# Problems whose data is pre-generated into _inputs/ (synthetic only).
_PREGEN_PROBLEMS = ["conditioned"]

# The runtime seed list is chosen by the harness from the global config and is
# NOT known at workspace-setup time (mid_edit only sees the task config). We
# pre-generate inputs for the task's declared seeds plus the standard seed set so
# that whichever seeds the harness runs are always covered. Generation is cheap
# and seed-deterministic, so any unused files are harmless.
_STANDARD_SEEDS = [42, 123, 456]
try:
    _cfg = json.loads((_TASK_DIR / "config.json").read_text())
    _SEEDS = sorted(set((_cfg.get("seeds") or []) + _STANDARD_SEEDS))
except Exception:
    _SEEDS = list(_STANDARD_SEEDS)


def _encode_input(problem, seed):
    # IMPORTANT: only the OBSERVABLE arrays (X_train, y_train, X_test) are written
    # into the workspace. The held-out test labels ``y_test`` are deliberately NOT
    # included -- otherwise the editable custom_vr.py could open this file, decode
    # y_test, and tune/emit the test labels directly. For 'conditioned' the driver
    # emits its test predictions and the host-side parser regenerates y_test (via
    # dgp.truth) and scores the same MSE, so y_test never enters the container.
    X_train, y_train, X_test, _y_test = dgp.gen_input(problem, seed)
    buf = io.BytesIO()
    np.savez(
        buf,
        X_train=np.ascontiguousarray(X_train),
        y_train=np.ascontiguousarray(y_train),
        X_test=np.ascontiguousarray(X_test),
    )
    return base64.b64encode(buf.getvalue()).decode("ascii")


OPS = [
    {
        "op": "create",
        "file": "opt-vr-bench/custom_vr.py",
        "content": _CUSTOM_PY,
    },
    {
        "op": "create",
        "file": "opt-vr-bench/vr_driver.py",
        "content": _DRIVER_PY,
    },
]

for _problem in _PREGEN_PROBLEMS:
    for _seed in _SEEDS:
        OPS.append({
            "op": "create",
            "file": f"opt-vr-bench/_inputs/{_problem}_seed{_seed}.npz.b64",
            "content": _encode_input(_problem, _seed),
        })
