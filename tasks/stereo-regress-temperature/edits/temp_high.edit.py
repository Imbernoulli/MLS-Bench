"""temp_high baseline for stereo-regress-temperature.

Reference: vendor/stereo-matching/baselines/temp_high.py
"""

_FILE = "stereo-matching/solution/temperature.py"

_CONTENT = 'def build_temperature():\n    return 8.0'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 24, "end_line": 28, "content": _CONTENT},
]
