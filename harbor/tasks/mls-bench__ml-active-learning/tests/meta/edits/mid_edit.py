"""Mid-edit operations for ml-active-learning task.

Creates the custom_sampling.py template file in the badge workspace
for the agent to modify with a novel query strategy, plus the runner script.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

_RUNNER_PATH = Path(__file__).parent / "run_al.py"
_RUNNER_PY = _RUNNER_PATH.read_text()

OPS = [
    {
        "op": "create",
        "file": "badge/query_strategies/custom_sampling.py",
        "content": _CUSTOM_PY,
    },
    {
        "op": "create",
        "file": "badge/run_al.py",
        "content": _RUNNER_PY,
    },
]
