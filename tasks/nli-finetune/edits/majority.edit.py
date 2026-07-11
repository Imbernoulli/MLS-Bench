"""Baseline edit for nli-finetune: majority."""

_FILE = "natural-language-inference/solution/finetune.py"

_CONTENT = '''def build_finetune() -> dict:
    return {"encoder": "majority"}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 11, "end_line": 14, "content": _CONTENT},
]
