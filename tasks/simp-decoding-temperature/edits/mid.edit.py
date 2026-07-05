"""Mid baseline: neutral sampling temperature (T=1.0, no reshaping).
Reference: vendor/text-simplification/baselines/temperature_mid.py
"""

_FILE = "text-simplification/solution/temperature.py"

_CONTENT = '''    return 1.0'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 40, "end_line": 40, "content": _CONTENT},
]
