"""Per-task scaffold overrides for the strict no-leak source-policy task."""

from pathlib import Path


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "vendor" / "abstractive-summarization").is_dir():
            return parent
    raise RuntimeError("could not locate MLS-Bench repo root from mid_edit.py")


_ABSTRACTIVE_SUMMARIZATION = _repo_root() / "vendor" / "abstractive-summarization"


def _read_vendor(rel: str) -> str:
    return (_ABSTRACTIVE_SUMMARIZATION / rel).read_text(encoding="utf-8")


OPS = [
    {
        "op": "create",
        "file": "abstractive-summarization/__init__.py",
        "content": "",
    },
    {
        "op": "create",
        "file": "abstractive-summarization/common.py",
        "content": _read_vendor("common.py"),
    },
    {
        "op": "create",
        "file": "abstractive-summarization/harness_source.py",
        "content": _read_vendor("harness_source.py"),
    },
    {
        "op": "create",
        "file": "abstractive-summarization/solution/source.py",
        "content": _read_vendor("solution/source.py"),
    },
]
