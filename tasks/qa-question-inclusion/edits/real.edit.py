"""Reference literal surface real."""

_FILE = 'extractive-qa/solution/question_inclusion.py'
_CONTENT = "def build_question_mode():\n    return 'real'"

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 5, "end_line": 6,
     "content": _CONTENT},
]
