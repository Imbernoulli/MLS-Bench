"""Real-but-weaker baseline: greedy T5 simplification decode.
Reference: vendor/text-simplification/baselines/policy_greedy.py
"""

_FILE = "text-simplification/solution/policy.py"

_CONTENT = '''    return "greedy"'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 34, "end_line": 35, "content": _CONTENT},
]
