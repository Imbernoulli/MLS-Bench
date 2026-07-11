"""Baseline `greedy` for summ-beam-width.
Reference: vendor/abstractive-summarization/baselines/
"""

_FILE = "abstractive-summarization/solution/beamwidth.py"

_CONTENT = '''def build_beam_width() -> int:
    # Greedy decoding.
    return 1'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 31, "end_line": 33, "content": _CONTENT},
]
