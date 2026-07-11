"""Baseline `cold` for summ-decoding-temperature.
Reference: vendor/abstractive-summarization/baselines/
"""

_FILE = "abstractive-summarization/solution/temperature.py"

_CONTENT = '''def build_temperature() -> float:
    # Temperature 0.3.
    return 0.3'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 31, "end_line": 33, "content": _CONTENT},
]
