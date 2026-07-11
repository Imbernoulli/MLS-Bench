"""Strong baseline: cold sampling temperature (sharpens towards the model's mode).
Reference: vendor/text-simplification/baselines/temperature_cold.py
"""

_FILE = "text-simplification/solution/temperature.py"

_CONTENT = '''    return 0.3'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 41, "end_line": 41, "content": _CONTENT},
]
