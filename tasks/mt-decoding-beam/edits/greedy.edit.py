"""Weak baseline: greedy decoding, no repetition control.
Reference: vendor/machine-translation/baselines/beam_greedy.py
"""

_FILE = "machine-translation/solution/beam.py"

_CONTENT = '''    return {"num_beams": 1, "no_repeat_ngram_size": 0}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 12, "end_line": 12, "content": _CONTENT},
]
