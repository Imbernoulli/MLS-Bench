"""Strong baseline: beam search width 5, no repetition block.
Reference: vendor/machine-translation/baselines/beam_tuned.py
"""

_FILE = "machine-translation/solution/beam.py"

_CONTENT = '''    return {"num_beams": 5, "no_repeat_ngram_size": 0}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 12, "end_line": 12, "content": _CONTENT},
]
