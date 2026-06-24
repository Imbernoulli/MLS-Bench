"""Mid-edit operations for ml-ensemble-boosting.

Creates custom_boosting.py (the agent's editable strategy + fixed runner) AND
pre-generates the standardized (X_train, y_train, X_test) triples the agent's
program loads at run time.

The dataset identity (sklearn dataset mapping), the loader, the deterministic
train/test split, the normalization, the test labels, and the metric live ONLY
in ``holdout/ml-ensemble-boosting/dgp.py`` (host-side, never bind-mounted). Here
we import it host-side and write ONLY the standardized (X_train, y_train, X_test)
plus the fixed pipeline hyperparameters into the workspace -- the agent never
sees which datasets are used nor y_test. Inputs are byte-identical to the
originals, so honest results are unchanged.
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
sys.path.insert(0, str(_PROJECT_ROOT / "holdout" / "ml-ensemble-boosting"))
import dgp  # host-only

_CUSTOM_PY = (_HERE.parent / "custom_template.py").read_text()

# The evaluated environments (test_cmd labels).
#
# The runtime seed list is chosen by the harness from the global config and is
# NOT known at workspace-setup time (mid_edit only sees the task config). We
# pre-generate inputs for the task's declared seeds plus the standard seed set
# so that whichever seeds the harness runs are always covered. Generation is
# cheap and seed-deterministic, so the unused files are harmless.
_STANDARD_SEEDS = [42, 123, 456]
try:
    _cfg = json.loads((_TASK_DIR / "config.json").read_text())
    _ENVS = [e.get("label") for e in _cfg.get("test_cmds", []) if e.get("label")]
    _SEEDS = sorted(set((_cfg.get("seeds") or []) + _STANDARD_SEEDS))
except Exception:
    _ENVS = ["breast_cancer", "diabetes", "california_housing"]
    _SEEDS = list(_STANDARD_SEEDS)


def _encode_input(env, seed):
    X_train, y_train, X_test, task_type, n_rounds, max_depth, lr = dgp.gen_input(
        env, seed=seed,
    )
    buf = io.BytesIO()
    np.savez(
        buf,
        X_train=np.ascontiguousarray(X_train, dtype=np.float64),
        y_train=np.ascontiguousarray(y_train, dtype=np.float64),
        X_test=np.ascontiguousarray(X_test, dtype=np.float64),
        task_type=np.array(task_type),
        n_rounds=np.int64(n_rounds),
        max_depth=np.int64(max_depth),
        learning_rate=np.float64(lr),
    )
    return base64.b64encode(buf.getvalue()).decode("ascii")


OPS = [
    {
        "op": "create",
        "file": "scikit-learn/custom_boosting.py",
        "content": _CUSTOM_PY,
    },
]

for _env in _ENVS:
    for _seed in _SEEDS:
        OPS.append({
            "op": "create",
            "file": f"scikit-learn/_boost_inputs/{_env}_seed{_seed}.npz.b64",
            "content": _encode_input(_env, _seed),
        })
