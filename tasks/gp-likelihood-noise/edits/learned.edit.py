_FILE = "gpytorch-gp/solution/likelihood_noise.py"

_CONTENT = '    return {"mode": "learned"}'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 5, "end_line": 5, "content": _CONTENT},
]
