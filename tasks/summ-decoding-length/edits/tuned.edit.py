"""Baseline `tuned` for summ-decoding-length.
Reference: vendor/abstractive-summarization/baselines/
"""

_FILE = "abstractive-summarization/solution/length.py"

_CONTENT = '''def build_length_config() -> dict:
    # Cross-domain window that fills XSum/CNN-DM/SAMSum targets.
    return {"min_length": 20, "max_length": 128, "length_penalty": 1.5}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 40, "end_line": 42, "content": _CONTENT},
]
