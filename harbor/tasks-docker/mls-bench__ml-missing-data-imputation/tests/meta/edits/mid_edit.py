"""Mid-edit operations for ml-missing-data-imputation.

Creates the custom imputation script in the scikit-learn workspace AND
pre-generates the masked input matrices the agent's program loads at run time.

The dataset loader, the true matrix, the missingness mask, and the labels live
ONLY in ``holdout/ml-missing-data-imputation/dgp.py`` (host-side, never
bind-mounted). Here we import it host-side and write ONLY the masked input
matrix (X with NaN at the masked entries) into the workspace — the agent never
sees which datasets are used for evaluation nor the held-out true values.
Inputs are byte-identical to the originals, so honest results are unchanged.
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
try:
    sys.path.insert(0, str(_PROJECT_ROOT / "holdout" / "ml-missing-data-imputation"))
    import dgp  # host-only
except Exception:
    # Held-out generator is not importable in every context that loads this
    # module (e.g. budget_check.py imports it only to read the editable-file
    # template). The per-input OPS below are then skipped; they are needed by
    # the inputgen, which runs where dgp IS staged.
    dgp = None

_CUSTOM_PY = (_HERE.parent / "custom_template.py").read_text()

_ENVS = ("breast_cancer", "wine", "california")
try:
    _cfg = json.loads((_TASK_DIR / "config.json").read_text())
    _SEEDS = sorted(set((_cfg.get("seeds") or []) + [42, 123, 456]))
except Exception:
    _SEEDS = [42, 123, 456]


def _encode_input(env, seed):
    X_missing = dgp.gen_input(env, seed=seed)
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(X_missing, dtype=np.float64))
    return base64.b64encode(buf.getvalue()).decode("ascii")


OPS = [
    {
        "op": "create",
        "file": "scikit-learn/custom_imputation.py",
        "content": _CUSTOM_PY,
    },
]

if dgp is not None:
    for _env in _ENVS:
        for _seed in _SEEDS:
            OPS.append({
                "op": "create",
                "file": f"scikit-learn/_impute_inputs/{dgp.opaque_label(_env)}_seed{_seed}.npy.b64",
                "content": _encode_input(_env, _seed),
            })
