"""Unmeasured no-demonstration candidate for full-protocol calibration.

Reference surface: vendor/code-generation-lab/solution/policy_fewshot.py
"""

_FILE = "code-generation-lab/solution/policy_fewshot.py"

_CONTENT = 'def fewshot():\n    return ""'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 10, "end_line": 11, "content": _CONTENT},
]
