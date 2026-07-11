"""Task-specific no-leak scaffold for nli-finetune."""
from pathlib import Path

_SCAFFOLD = Path(__file__).resolve().parent / "scaffold"


def _read(rel: str) -> str:
    return (_SCAFFOLD / rel).read_text()


OPS = [
    {
        "op": "create",
        "file": "natural-language-inference/__init__.py",
        "content": _read("natural-language-inference/__init__.py"),
    },
    {
        "op": "create",
        "file": "natural-language-inference/solution/finetune.py",
        "content": _read("natural-language-inference/solution/finetune.py"),
    },
]
