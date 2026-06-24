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

try:
    _cfg = json.loads((_TASK_DIR / "config.json").read_text())
    _SEEDS = _cfg.get("seeds") or [42]
except Exception:
    _SEEDS = [42]


def _encode_input(problem, seed):
    X_train, y_train, X_test, y_test = dgp.gen_input(problem, seed)
    buf = io.BytesIO()
    np.savez(
        buf,
        X_train=np.ascontiguousarray(X_train),
        y_train=np.ascontiguousarray(y_train),
        X_test=np.ascontiguousarray(X_test),
        y_test=np.ascontiguousarray(y_test),
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
