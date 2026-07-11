"""Reference literal surface question_first."""

_FILE = 'extractive-qa/solution/encoding_order.py'
_CONTENT = "def build_encoding_order():\n    return 'question_first'"

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 5, "end_line": 6,
     "content": _CONTENT},
]
