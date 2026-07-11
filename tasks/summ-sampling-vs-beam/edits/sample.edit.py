"""Baseline `sample` for summ-sampling-vs-beam.
Reference: vendor/abstractive-summarization/baselines/
"""

_FILE = "abstractive-summarization/solution/sampling.py"

_CONTENT = '''def build_decode_strategy() -> dict:
    # Nucleus sampling (weak for ROUGE).
    return {"strategy": "sample", "top_p": 0.95, "top_k": 0, "temperature": 1.0}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 34, "end_line": 36, "content": _CONTENT},
]
