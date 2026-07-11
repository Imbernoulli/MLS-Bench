"""Strong baseline: beam width 5, no-repeat-3gram block (tuned decode).
Reference: vendor/text-simplification/baselines/beam_tuned.py
"""

_FILE = "text-simplification/solution/beam.py"

_CONTENT = '''    return {"num_beams": 5, "no_repeat_ngram_size": 3, "repetition_penalty": 1.0}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 41, "end_line": 41, "content": _CONTENT},
]
