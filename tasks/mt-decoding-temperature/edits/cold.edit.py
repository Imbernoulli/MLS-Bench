"""Baseline cold for mt-decoding-temperature.
Reference: vendor/machine-translation/solution/temperature.py
"""

_FILE = "machine-translation/solution/temperature.py"

_CONTENT = '''    return 0.3'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 11, "end_line": 11, "content": _CONTENT},
]
