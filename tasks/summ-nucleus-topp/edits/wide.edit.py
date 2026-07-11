"""Baseline `wide` for summ-nucleus-topp.
Reference: vendor/abstractive-summarization/baselines/
"""

_FILE = "abstractive-summarization/solution/topp.py"

_CONTENT = '''    # Wide nucleus candidate.
    return 1.0'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 33, "end_line": 33, "content": _CONTENT},
]
