"""Mid-edit operations for the ml-symbolic-regression task.

Creates custom_sr.py (the agent's editable GP file) AND pre-generates the
``(X_train, y_train, X_test)`` triples the agent's program loads at run time.

The benchmark TARGET FUNCTIONS (closed forms), the data generator, and the R2
metric live ONLY in ``holdout/ml-symbolic-regression/dgp.py`` (host-side, never
bind-mounted into the agent container). Here we import it host-side and write
ONLY training SAMPLES (X_train, y_train) plus the test inputs (X_test) into the
workspace, keyed by an opaque, salted, name-free token. The agent never sees
which benchmark is used, the closed-form target, or the test labels. The inputs
(X_train, y_train, X_test) are byte-identical to the originals, so honest
results are unchanged.
"""

import io
import json
import base64
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_TASK_DIR = _HERE.parents[1]
_PROJECT_ROOT = _HERE.parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "holdout" / "ml-symbolic-regression"))
import dgp  # host-only

_CUSTOM_PY = (_HERE.parent / "custom_template.py").read_text()

try:
    _cfg = json.loads((_TASK_DIR / "config.json").read_text())
    _BENCHMARKS = [e.get("label") for e in _cfg.get("test_cmds", []) if e.get("label")]
    _SEEDS = _cfg.get("seeds") or [42]
except Exception:
    _BENCHMARKS, _SEEDS = ["nguyen7", "nguyen10", "koza3"], [42, 123, 456]


def _encode_input(benchmark, seed):
    X_train, y_train, X_test = dgp.gen_input(benchmark, seed=seed)
    buf = io.BytesIO()
    np.savez(
        buf,
        X_train=np.ascontiguousarray(X_train, dtype=np.float64),
        y_train=np.ascontiguousarray(y_train, dtype=np.float64),
        X_test=np.ascontiguousarray(X_test, dtype=np.float64),
    )
    return base64.b64encode(buf.getvalue()).decode("ascii")


OPS = [
    {
        "op": "create",
        "file": "gplearn/custom_sr.py",
        "content": _CUSTOM_PY,
    },
]

for _bench in _BENCHMARKS:
    _tid = dgp.task_id(_bench)
    for _seed in _SEEDS:
        OPS.append({
            "op": "create",
            "file": f"gplearn/_sr_inputs/{_tid}_seed{_seed}.npz.b64",
            "content": _encode_input(_bench, _seed),
        })
