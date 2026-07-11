"""Baseline `beam` for summ-beam-width.
Reference: vendor/abstractive-summarization/baselines/
"""

_FILE = "abstractive-summarization/solution/beamwidth.py"

_CONTENT = '''def build_beam_width() -> int:
    # Beam search (strong).
    return 4'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 31, "end_line": 33, "content": _CONTENT},
]
