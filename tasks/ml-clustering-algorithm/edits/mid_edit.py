"""Mid-edit operations for ml-clustering-algorithm.

Creates the custom clustering script in the scikit-learn workspace AND
pre-generates the standardized feature matrices the agent's program loads.

The dataset generator, the true labels, the dataset identity, and the metrics
live ONLY in ``holdout/ml-clustering-algorithm/dgp.py`` (host-side, never
bind-mounted). Here we import it host-side and write ONLY the standardized
matrix + the (metadata) number of clusters into the workspace — the agent never
sees which datasets are used nor the true cluster assignments. Inputs are
byte-identical to the originals, so honest results are unchanged.
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
sys.path.insert(0, str(_PROJECT_ROOT / "holdout" / "ml-clustering-algorithm"))
import dgp  # host-only

_CUSTOM_PY = (_HERE.parent / "custom_template.py").read_text()

# The evaluated environments (test_cmd labels).
try:
    _cfg = json.loads((_TASK_DIR / "config.json").read_text())
    _ENVS = [e.get("label") for e in _cfg.get("test_cmds", []) if e.get("label")]
    _SEEDS = sorted(set((_cfg.get("seeds") or []) + [42, 123, 456]))
except Exception:
    _ENVS, _SEEDS = ["blobs", "moons", "digits"], [42]


def _encode_input(env, seed):
    X, n_clusters = dgp.gen_input(env, seed=seed)
    buf = io.BytesIO()
    np.savez(buf, X=np.ascontiguousarray(X, dtype=np.float64),
             n_clusters=np.int64(n_clusters))
    return base64.b64encode(buf.getvalue()).decode("ascii")


OPS = [
    {
        "op": "create",
        "file": "scikit-learn/custom_clustering.py",
        "content": _CUSTOM_PY,
    },
]

for _env in _ENVS:
    for _seed in _SEEDS:
        OPS.append({
            "op": "create",
            "file": f"scikit-learn/_cluster_inputs/{_env}_seed{_seed}.npz.b64",
            "content": _encode_input(_env, _seed),
        })
