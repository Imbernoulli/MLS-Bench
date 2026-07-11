"""Baseline edit for nli-finetune: finetune."""

_FILE = "natural-language-inference/solution/finetune.py"

_CONTENT = '''def build_finetune() -> dict:
    return {"encoder": "finetune"}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 11, "end_line": 14, "content": _CONTENT},
]
