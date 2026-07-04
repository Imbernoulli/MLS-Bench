"""Mid-edit for optimization-nas: create the editable search scaffold and
materialize ONLY the per-dataset VALIDATION table into the workspace.

The full NAS-Bench-201 pickle (which carries the held-out TEST accuracies of
all 15625 architectures) never enters the agent's workspace: a create-op below
overwrites the package's pickle path with a placeholder note, and the runner
instead loads a per-(dataset, seed) VALIDATION table materialized here from
the host-side provider ``holdout/optimization-nas/dgp.py``. The held-out TEST
accuracy of the final architecture is joined by the task parser OUTSIDE the
agent's process (the runner's ``FINAL_ARCH`` report).

In the Harbor verifier this module is re-imported for every evaluation by
``apply.py`` with ENV/SEED exported, so only the active run's table is
materialized — and the runner deletes it before any editable code executes
(MLSBENCH_EPHEMERAL_INPUTS=1 in the eval scripts). Natively (no ENV at
workspace-setup time) every (dataset, seed) combination is materialized once
and kept, so repeated tests in the same workspace keep working.
"""

import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_TASK_DIR = _HERE.parents[1]            # tasks/optimization-nas
_PROJECT_ROOT = _HERE.parents[3]        # repo root (or the _inputgen root in Harbor)
_HOLDOUT_DIR = _PROJECT_ROOT / "holdout" / "optimization-nas"
sys.path.insert(0, str(_HOLDOUT_DIR))
import dgp  # host-/verifier-only table provider (never shipped to the agent)

_TEMPLATE_PATH = _HERE.parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

# Match the harness seed selection: task seeds if present, plus the default.
try:
    _cfg = json.loads((_TASK_DIR / "config.json").read_text())
    _SEEDS = sorted(set((_cfg.get("seeds") or []) + [42]))
except Exception:
    _SEEDS = [0, 1, 2, 3, 4, 42]

_ENV = os.environ.get("ENV")
_SEED = os.environ.get("SEED")
if _ENV in dgp.DATASET_MAP and _SEED is not None:
    # Harbor eval-time materialization: just the active run's table.
    _COMBOS = [(_ENV, int(_SEED))]
else:
    _COMBOS = [(_env, _seed) for _env in dgp.DATASET_MAP for _seed in _SEEDS]

_PLACEHOLDER = (
    "NAS-Bench-201 accuracy tables are not available in the agent workspace.\n"
    "The harness materializes the validation table for the active dataset at\n"
    "evaluation time; the held-out test accuracy of the final architecture is\n"
    "joined outside this process after the search reports FINAL_ARCH.\n"
)

# ── Mid-edit operations ──────────────────────────────────────────────

OPS = [
    {
        "op": "create",
        "file": "naslib/custom_nas_search.py",
        "content": _CUSTOM_PY,
    },
    {
        # Overwrite the package's full benchmark pickle (val+test accuracies
        # for every architecture) with a note; the runner no longer reads it.
        "op": "create",
        "file": "naslib/naslib/data/nb201_all.pickle",
        "content": _PLACEHOLDER,
    },
]

for _env, _seed in _COMBOS:
    OPS.append({
        "op": "create",
        "file": f"naslib/naslib/data/nb201_tables_{_env}_s{_seed}.json",
        "content": json.dumps({"val": dgp.val_table(_env)}),
    })
