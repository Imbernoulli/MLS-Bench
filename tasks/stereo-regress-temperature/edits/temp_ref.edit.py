"""temp_ref baseline for stereo-regress-temperature.

Reference: vendor/stereo-matching/baselines/temp_ref.py
"""

_FILE = "stereo-matching/solution/temperature.py"

_CONTENT = 'def build_temperature():\n    return 1.0'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 24, "end_line": 28, "content": _CONTENT},
]
