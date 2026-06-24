"""Mid-edit operations for the causal-treatment-effect task.

Applied to the scikit-learn workspace after pre_edit, before the agent starts.

Two responsibilities:
  1. Create ``custom_cate.py`` — the agent's editable algorithm file.
  2. Pre-generate the observational inputs (X, T, Y) that the agent's program
     loads at run time.

The data-generating process and the true treatment effect ``tau`` live ONLY in
``holdout/causal-treatment-effect/dgp.py`` (host-side, never bind-mounted into
the agent container). Here we import it host-side, generate the inputs
deterministically from the same DGP + seeds as before, and write ONLY (X, T, Y)
into the workspace. ``tau`` is never written anywhere the container can read it,
so the agent's process cannot reach the answer; honest results are unchanged
because the data is byte-identical.

Inputs are written as base64-encoded ``.npz`` text files (the create op writes
text), decoded by ``custom_cate.py``'s ``load_inputs``. The encoded bytes are
computed at setup time, so this file stays small and nothing large is committed.
"""

import io
import json
import base64
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_TASK_DIR = _HERE.parents[1]            # tasks/causal-treatment-effect
_PROJECT_ROOT = _HERE.parents[3]        # repo root
_HOLDOUT_DIR = _PROJECT_ROOT / "holdout" / "causal-treatment-effect"
sys.path.insert(0, str(_HOLDOUT_DIR))
import dgp  # host-only DGP + metrics (NOT shipped into the container)

_TEMPLATE_PATH = _HERE.parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

# Match the harness seed selection: task seeds if present, else default [42].
try:
    _cfg = json.loads((_TASK_DIR / "config.json").read_text())
    _SEEDS = _cfg.get("seeds") or [42]
except Exception:
    _SEEDS = [42]
_N_REPS = 10


def _encode_inputs(dataset, data_seed):
    X, T, Y = dgp.gen_inputs(dataset, data_seed)
    buf = io.BytesIO()
    np.savez(buf, X=X, T=T, Y=Y)
    return base64.b64encode(buf.getvalue()).decode("ascii")


OPS = [
    {
        "op": "create",
        "file": "scikit-learn/custom_cate.py",
        "content": _CUSTOM_PY,
    },
]

for _dataset in dgp.DATASETS:
    for _seed in _SEEDS:
        for _rep in range(_N_REPS):
            _data_seed = _seed + _rep * 1000
            OPS.append({
                "op": "create",
                "file": f"scikit-learn/_cate_inputs/{_dataset}_seed{_data_seed}.npz.b64",
                "content": _encode_inputs(_dataset, _data_seed),
            })
