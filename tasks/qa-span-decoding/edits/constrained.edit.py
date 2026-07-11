"""Reference literal surface constrained."""

_FILE = 'extractive-qa/solution/span_decoding.py'
_CONTENT = "def build_decoder():\n    return 'constrained'"

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 5, "end_line": 6,
     "content": _CONTENT},
]
