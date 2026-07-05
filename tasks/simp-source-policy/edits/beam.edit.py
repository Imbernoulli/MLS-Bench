"""Strong (SOTA-scale) baseline: tuned-beam T5 simplification decode.
Reference: vendor/text-simplification/baselines/policy_beam.py
"""

_FILE = "text-simplification/solution/policy.py"

_CONTENT = '''    return "beam"'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 39, "end_line": 39, "content": _CONTENT},
]
