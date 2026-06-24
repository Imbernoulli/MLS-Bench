"""Mid-edit operations for causal-discovery-discrete.

Creates the bench/ evaluation scaffold inside the causal-bnlearn package workspace
AND pre-generates the integer-encoded observational samples X the driver loads at
run time:

  bench/run_eval.py                 — FIXED driver: loads pre-generated X, calls
                                      run_causal_discovery(X), serializes the
                                      estimated graph, prints CAUSAL_PRED.
  bench/custom_algorithm.py         — agent-editable algorithm entry point.
  bench/_causal_inputs/<label>_seed<seed>.npz.b64 — pre-generated X ONLY (no
                                      ground-truth DAG, no network name).

The eval network identity (which bnlearn network each label maps to), the
ground-truth DAG construction (``get_example_model``), and the metric all live in
``holdout/causal-discovery-discrete/dgp.py`` — a host-only module that is NOT
bind-mounted into the agent container. Here we import it host-side and write ONLY
the samples X into the workspace, so the agent never sees the network name and can
neither read nor regenerate the answer. X is byte-identical to the data the old
in-container ``data_gen.py`` produced, so honest results are unchanged.
"""

import base64
import io
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_TASK_DIR = _HERE.parents[1]
_PROJECT_ROOT = _HERE.parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "holdout" / "causal-discovery-discrete"))
import dgp  # host-only (never bind-mounted into the agent container)

try:
    _cfg = json.loads((_TASK_DIR / "config.json").read_text())
    _LABELS = [e.get("label") for e in _cfg.get("test_cmds", []) if e.get("label")]
    _SEEDS = _cfg.get("seeds") or [42]
except Exception:
    _LABELS = ["Cancer", "Child", "Alarm", "Hailfinder", "Win95pts"]
    _SEEDS = [42]


def _encode_input(label, seed):
    X = dgp.gen_input(label, seed=seed)
    buf = io.BytesIO()
    np.savez(buf, X=np.ascontiguousarray(X, dtype=np.int64))
    return base64.b64encode(buf.getvalue()).decode("ascii")


OPS = [
    {
        "op": "create",
        "file": "causal-bnlearn/bench/run_eval.py",
        "content": (_HERE.parent / "run_eval_template.py").read_text(),
    },
    {
        "op": "create",
        "file": "causal-bnlearn/bench/custom_algorithm.py",
        "content": (_HERE.parent / "custom_template.py").read_text(),
    },
]

for _label in _LABELS:
    for _seed in _SEEDS:
        OPS.append({
            "op": "create",
            "file": f"causal-bnlearn/bench/_causal_inputs/{_label}_seed{_seed}.npz.b64",
            "content": _encode_input(_label, _seed),
        })
