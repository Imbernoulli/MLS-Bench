"""Weak baseline: t5-small checkpoint, broader wiki_auto_asset_turk fine-tune mix.
Reference: vendor/text-simplification/baselines/capacity_small_wikiauto.py
"""

_FILE = "text-simplification/solution/capacity.py"

_CONTENT = '''    return "small_wikiauto"'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 46, "end_line": 46, "content": _CONTENT},
]
