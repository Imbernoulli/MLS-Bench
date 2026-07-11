"""Unmeasured identity-extraction candidate for full-protocol calibration.

Reference surface: vendor/code-generation-lab/solution/policy_extract.py
"""

_FILE = "code-generation-lab/solution/policy_extract.py"

_CONTENT = 'def extract(raw_text, entry_point):\n    return raw_text'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 10, "end_line": 11, "content": _CONTENT},
]
