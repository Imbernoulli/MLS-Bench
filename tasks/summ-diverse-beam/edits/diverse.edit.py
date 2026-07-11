"""Baseline `diverse` for summ-diverse-beam.
Reference: vendor/abstractive-summarization/baselines/
"""

_FILE = "abstractive-summarization/solution/diverse.py"

_CONTENT = '''def build_diverse_config() -> dict:
    # Diverse beam search (4 groups) -> lower single-best ROUGE (weak).
    return {"num_beams": 4, "num_beam_groups": 4, "diversity_penalty": 1.0}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 33, "end_line": 35, "content": _CONTENT},
]
