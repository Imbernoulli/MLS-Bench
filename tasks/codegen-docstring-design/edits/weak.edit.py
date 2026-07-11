"""Unmeasured minimal-prompt candidate for full-protocol calibration.

Reference surface: vendor/code-generation-lab/solution/policy_docstring.py
"""

_FILE = "code-generation-lab/solution/policy_docstring.py"

_CONTENT = 'def build_prompt(problem):\n    return problem["prompt"]'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 10, "end_line": 11, "content": _CONTENT},
]
