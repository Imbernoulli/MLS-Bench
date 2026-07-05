"""Weak baseline: greedy decoding, no repetition control (under-searches).
Reference: vendor/text-simplification/baselines/beam_greedy.py
"""

_FILE = "text-simplification/solution/beam.py"

_CONTENT = '''    return {"num_beams": 1, "no_repeat_ngram_size": 0, "repetition_penalty": 1.0}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 39, "end_line": 39, "content": _CONTENT},
]
