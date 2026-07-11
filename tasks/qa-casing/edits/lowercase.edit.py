"""Reference literal surface lowercase."""

_FILE = 'extractive-qa/solution/casing.py'
_CONTENT = "def build_casing():\n    return 'lowercase'"

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 5, "end_line": 6,
     "content": _CONTENT},
]
