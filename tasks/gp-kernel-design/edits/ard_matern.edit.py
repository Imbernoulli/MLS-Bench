_FILE = "gpytorch-gp/solution/kernel_design.py"

_CONTENT = '''    return {
        "kernel": "matern", "ard": True, "nu": 2.5, "mean": "constant"
    }'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 5, "end_line": 5, "content": _CONTENT},
]
