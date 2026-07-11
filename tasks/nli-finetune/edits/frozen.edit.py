"""Baseline edit for nli-finetune: frozen."""

_FILE = "natural-language-inference/solution/finetune.py"

_CONTENT = '''def build_finetune() -> dict:
    return {"encoder": "frozen"}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 11, "end_line": 14, "content": _CONTENT},
]
