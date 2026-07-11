"""Baseline mid for mt-diverse-beam.
Reference: vendor/machine-translation/solution/divbeam.py
"""

_FILE = "machine-translation/solution/divbeam.py"

_CONTENT = '''    return {"num_beam_groups": 4, "diversity_penalty": 0.5}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 11, "end_line": 11, "content": _CONTENT},
]
