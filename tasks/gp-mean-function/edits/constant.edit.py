_FILE = "gpytorch-gp/solution/mean_function.py"

_CONTENT = '    return {"mean": "constant"}'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 5, "end_line": 5, "content": _CONTENT},
]
