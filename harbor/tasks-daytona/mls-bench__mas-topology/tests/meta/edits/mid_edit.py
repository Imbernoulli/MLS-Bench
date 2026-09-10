"""Mid-edit operations for the mas-topology task.

Applied to the chatdev-macnet workspace after pre_edit, before the agent starts.

Creates:
1. custom_topology.py -- the agent's editable topology function
2. run_with_topology.py -- patched entry point that uses custom_topology
3. run_humaneval.py -- HumanEval benchmark runner
4. run_srdd.py -- SRDD benchmark runner
5. srdd_queries.json -- SRDD query set (20 curated prompts)
"""

from pathlib import Path

_DIR = Path(__file__).parent
_TASK_DIR = _DIR.parent

_CUSTOM_PY = (_DIR / "custom_template.py").read_text()
_RUN_WITH_TOPOLOGY_PY = (_DIR / "run_with_topology.py").read_text()
_RUN_HUMANEVAL_PY = (_DIR / "run_humaneval.py").read_text()
_RUN_SRDD_PY = (_DIR / "run_srdd.py").read_text()
_SRDD_QUERIES = (_TASK_DIR / "srdd_queries.json").read_text()

# ── Mid-edit operations ──────────────────────────────────────────────

OPS = [
    {
        "op": "create",
        "file": "chatdev-macnet/custom_topology.py",
        "content": _CUSTOM_PY,
    },
    {
        "op": "create",
        "file": "chatdev-macnet/run_with_topology.py",
        "content": _RUN_WITH_TOPOLOGY_PY,
    },
    {
        "op": "create",
        "file": "chatdev-macnet/run_humaneval.py",
        "content": _RUN_HUMANEVAL_PY,
    },
    {
        "op": "create",
        "file": "chatdev-macnet/run_srdd.py",
        "content": _RUN_SRDD_PY,
    },
    {
        "op": "create",
        "file": "chatdev-macnet/srdd_queries.json",
        "content": _SRDD_QUERIES,
    },
]
