"""Task-specific scaffold for reid-metric-loss."""

from pathlib import Path


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "vendor" / "torchreid-reid").is_dir():
            return parent
    raise RuntimeError("could not locate MLS-Bench repo root from mid_edit.py")


_VENDOR = _repo_root() / "vendor" / "torchreid-reid"


def _read(rel: str) -> str:
    return (_VENDOR / rel).read_text(encoding="utf-8")


def _create(rel: str, content: str | None = None) -> dict:
    if content is None:
        content = _read(rel)
    return {
        "op": "create",
        "file": f"torchreid-reid/{rel}",
        "content": content,
    }


OPS = [
    {
        "op": "create",
        "file": "torchreid-reid/__init__.py",
        "content": "",
    },
    _create("common.py"),
    _create("harness_loss.py"),
    {
        "op": "create",
        "file": "torchreid-reid/solution/__init__.py",
        "content": "",
    },
    _create("solution/loss.py"),
]

for path in sorted((_VENDOR / "torchreid").rglob("*.py")):
    rel = path.relative_to(_VENDOR).as_posix()
    OPS.append(_create(rel))
