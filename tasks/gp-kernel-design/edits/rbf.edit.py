_FILE = "gpytorch-gp/solution/kernel_design.py"

_CONTENT = '    return {"kernel": "rbf", "ard": False, "mean": "constant"}'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 5, "end_line": 5, "content": _CONTENT},
]
