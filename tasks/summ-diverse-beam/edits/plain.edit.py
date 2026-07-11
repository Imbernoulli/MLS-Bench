"""Baseline `plain` for summ-diverse-beam.
Reference: vendor/abstractive-summarization/baselines/
"""

_FILE = "abstractive-summarization/solution/diverse.py"

_CONTENT = '''def build_diverse_config() -> dict:
    # One-group beam decoding.
    return {"num_beams": 4, "num_beam_groups": 1, "diversity_penalty": 0.0}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 33, "end_line": 35, "content": _CONTENT},
]
