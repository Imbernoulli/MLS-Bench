_FILE = "gpytorch-gp/solution/inducing.py"

_CONTENT = '    return {"method": "random", "count": 16}'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 5, "end_line": 5, "content": _CONTENT},
]
