"""Baseline topp for mt-sampling-vs-beam.
Reference: vendor/machine-translation/solution/sampling.py
"""

_FILE = "machine-translation/solution/sampling.py"

_CONTENT = '''    return "topp"'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 11, "end_line": 11, "content": _CONTENT},
]
