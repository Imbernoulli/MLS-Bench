"""Mid-edit operations for the ml-anomaly-detection task.

Creates custom_anomaly.py (the agent's editable detector) AND pre-generates the
standardized (X_train, X_test) pairs the agent's program loads at run time.

The dataset identity (adbench file mapping), the loader, the train/test split,
the test labels, and the metrics live ONLY in
``holdout/ml-anomaly-detection/dgp.py`` (host-side, never bind-mounted). Here we
import it host-side and write ONLY the standardized (train, test) features into
the workspace — the agent never sees which datasets are used nor y_test. Inputs
are byte-identical to the originals, so honest results are unchanged.
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
sys.path.insert(0, str(_PROJECT_ROOT / "holdout" / "ml-anomaly-detection"))
import dgp  # host-only

_CUSTOM_PY = (_HERE.parent / "custom_template.py").read_text()

try:
    _cfg = json.loads((_TASK_DIR / "config.json").read_text())
    _ENVS = [e.get("label") for e in _cfg.get("test_cmds", []) if e.get("label")]
    _SEEDS = sorted(set((_cfg.get("seeds") or []) + [42, 123, 456]))
except Exception:
    _ENVS, _SEEDS = ["cardio", "thyroid", "satellite", "shuttle"], [42]


def _encode_input(env, seed):
    X_train, X_test = dgp.gen_input(env, seed=seed)
    buf = io.BytesIO()
    np.savez(buf, X_train=np.ascontiguousarray(X_train, dtype=np.float64),
             X_test=np.ascontiguousarray(X_test, dtype=np.float64))
    return base64.b64encode(buf.getvalue()).decode("ascii")


OPS = [
    {
        "op": "create",
        "file": "scikit-learn/custom_anomaly.py",
        "content": _CUSTOM_PY,
    },
]

for _env in _ENVS:
    for _seed in _SEEDS:
        OPS.append({
            "op": "create",
            "file": f"scikit-learn/_anomaly_inputs/{dgp.opaque_label(_env)}_seed{_seed}.npz.b64",
            "content": _encode_input(_env, _seed),
        })
