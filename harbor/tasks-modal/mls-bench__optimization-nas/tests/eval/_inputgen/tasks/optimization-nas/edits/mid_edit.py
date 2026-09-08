"""Mid-edit for optimization-nas: create the editable search scaffold and
materialize ONLY the per-dataset VALIDATION table into the workspace.

The full NAS-Bench-201 pickle (which carries the held-out TEST accuracies of
all 15625 architectures) never enters the agent's workspace: a create-op below
overwrites the package's pickle path with a placeholder note, and the runner
instead loads a per-(dataset, seed) VALIDATION table materialized here from
the host-side provider ``holdout/optimization-nas/dgp.py``. The held-out TEST
accuracy of the final architecture is joined by the task parser OUTSIDE the
agent's process (the runner's ``FINAL_ARCH`` report).

This module is re-run for every evaluation with ENV/SEED exported — by
``apply.py`` in the Harbor verifier and by the harness's host-side
``mlsbench.agent.input_stager`` natively (config.json ``"ephemeral_inputs"``)
— so only the active run's table is materialized, and the runner deletes it
after loading, before any editable code executes (MLSBENCH_EPHEMERAL_INPUTS=1
in the eval scripts). At workspace-setup time (no ENV/SEED) NO tables are
staged (only the editable scaffold and the pickle placeholder) — matching
Harbor's agent-session _scaffold. Tables staged at setup would linger
unconsumed in the shared workspace, readable by editable code during other
runs' evaluations (sibling dataset/seed tables are a correlated proxy for
the budgeted oracle); each evaluation re-stages its own table instead.
"""

import importlib.util as _ilu
import json
import os
from pathlib import Path

_HERE = Path(__file__).resolve()
_TASK_DIR = _HERE.parents[1]            # tasks/optimization-nas
_PROJECT_ROOT = _HERE.parents[3]        # repo root (or the _inputgen root in Harbor)
_HOLDOUT_DIR = _PROJECT_ROOT / "holdout" / "optimization-nas"
# Load the host-/verifier-only table provider under a UNIQUE module name (not a
# bare ``import dgp``): several tasks ship a holdout/<task>/dgp.py, and any
# context that execs multiple tasks' mid_edits in one process (e.g. the website
# exporter scripts/build_site_data.py) would otherwise collide on
# sys.modules["dgp"]. Native setup and the Harbor inputgen exec one task per
# process, so this is belt-and-suspenders there.
_dgp_spec = _ilu.spec_from_file_location("optimization_nas_dgp", str(_HOLDOUT_DIR / "dgp.py"))
dgp = _ilu.module_from_spec(_dgp_spec)
_dgp_spec.loader.exec_module(dgp)

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
# The native harness exports the test label as ENV ("CIFAR-10"); the Harbor
# eval scripts export the dataset key directly ("cifar10"). Accept both so
# eval-time materialization stays selective — staging ONLY the active run's
# table also means no sibling dataset/seed table (a correlated proxy for the
# budgeted oracle) is ever on disk during an evaluation.
_ENV = {
    "CIFAR-10": "cifar10",
    "CIFAR-100": "cifar100",
    "ImageNet16-120": "imagenet16",
}.get(_ENV, _ENV)
if _ENV in dgp.DATASET_MAP and _SEED is not None:
    # Eval-time materialization (Harbor apply.py / native input_stager):
    # just the active run's table.
    _COMBOS = [(_ENV, int(_SEED))]
else:
    # Workspace setup (no ENV/SEED) or an unrecognized label: stage NO
    # tables — only the editable scaffold and the pickle placeholder below.
    # Tables staged here would linger unconsumed in the shared workspace,
    # readable by editable code during other runs' evaluations.
    _COMBOS = []

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
