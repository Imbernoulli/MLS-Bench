_FILE = "gpytorch-gp/solution/exact_lr.py"

_CONTENT = '    return {"learning_rate": 0.1}'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 5, "end_line": 5, "content": _CONTENT},
]
