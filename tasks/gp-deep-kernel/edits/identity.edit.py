_FILE = "gpytorch-gp/solution/deep_kernel.py"

_CONTENT = '    return {"extractor": "identity"}'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 5, "end_line": 5, "content": _CONTENT},
]
