"""Mid-edit operations for the optimization-multi-objective task.

Two responsibilities:
  1. Create ``deap/custom_moea.py`` — the agent's editable strategy file.
  2. Pre-generate, per (problem, seed), the opaque problem SPEC the agent's
     program loads at run time (numeric configuration + a marshalled black-box
     objective evaluator). The spec carries NO problem name and NO Pareto front.

The benchmark problem identity (ZDT/DTLZ), the analytic true Pareto fronts, and
the IGD / hypervolume / spread metrics live ONLY in
``holdout/optimization-multi-objective/dgp.py`` (host-side, never bind-mounted
into the agent container). Here we import it host-side and write ONLY the opaque
spec into the workspace, keyed by the per-problem alias (``p0``..``p3``) that the
run scripts pass via ``ENV``. The agent never sees which problem is used nor its
true front. The objective evaluation is byte-identical to the original, so
honest results are unchanged.

Specs are written as base64-encoded JSON text files (the create op writes text),
decoded by ``custom_moea.py``'s ``_load_spec``. The encoded bytes are computed at
setup time, so this file stays small and nothing large is committed.
"""

import base64
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_TASK_DIR = _HERE.parents[1]            # tasks/optimization-multi-objective
_PROJECT_ROOT = _HERE.parents[3]        # repo root
_HOLDOUT_DIR = _PROJECT_ROOT / "holdout" / "optimization-multi-objective"
sys.path.insert(0, str(_HOLDOUT_DIR))
import dgp  # host-only problem defs + fronts + metrics (NOT shipped into the container)

_TEMPLATE_PATH = _HERE.parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

# The runtime seed list is chosen by the harness from the global config and is
# NOT known at workspace-setup time (mid_edit only sees the task config). We
# pre-generate specs for the task's declared seeds plus the standard seed set so
# that whichever seeds the harness runs are always covered.
_STANDARD_SEEDS = [42, 123, 456]
try:
    _cfg = json.loads((_TASK_DIR / "config.json").read_text())
    _SEEDS = sorted(set((_cfg.get("seeds") or []) + _STANDARD_SEEDS))
    # Problem names come from the test_cmd labels (host-side only).
    _PROBLEMS = [e.get("label") for e in _cfg.get("test_cmds", []) if e.get("label")]
except Exception:
    _SEEDS = list(_STANDARD_SEEDS)
    _PROBLEMS = ["zdt1", "zdt3", "dtlz2", "dtlz1"]


def _encode_spec(problem, seed):
    spec = dgp.gen_problem(problem, seed=seed)
    return base64.b64encode(json.dumps(spec).encode("utf-8")).decode("ascii")


OPS = [
    {
        "op": "create",
        "file": "deap/custom_moea.py",
        "content": _CUSTOM_PY,
    },
]

for _problem in _PROBLEMS:
    _alias = dgp.ALIASES[_problem]
    for _seed in _SEEDS:
        OPS.append({
            "op": "create",
            "file": f"deap/_moea_specs/{_alias}_seed{_seed}.json.b64",
            "content": _encode_spec(_problem, _seed),
        })
