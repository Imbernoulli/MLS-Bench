#!/usr/bin/env python3
"""Ratchet: a parser that runs on a Python 3.8 image must defer its annotations.

Some packages still build on images whose python is 3.8 (Ubuntu 20.04, and
NGC's ``pytorch:22.04-py3``), and the Harbor verifier runs the task's ``parser.py``
with whichever interpreter owns the package's ML stack -- on those images that
is 3.8. PEP 585 builtin generics in an *evaluated* position, e.g.

    def _parse_eval_scores(self, output: str, cmd_label: str) -> tuple[str, dict]:

are evaluated when the ``def`` executes, so on 3.8 the parser dies at import
with ``TypeError: 'type' object is not subscriptable`` -- before it ever sees
any output. The task then scores 0 with nothing in the log pointing at the
parser. ``from __future__ import annotations`` makes them strings and costs
nothing on newer interpreters.

Verified against real 3.8: without the import these parsers fail to import on
``python:3.8-slim`` and on the shipped ``mlsbench-harbor-humanoid-gym`` image
(conda python 3.8.13); with it they import cleanly.

Only tasks on a 3.8 image are checked. The same annotations are harmless on the
pytorch/pytorch:2.x and python:3.10+ images the other tasks use, so requiring
the import everywhere would be noise.

Usage:
    python tests/test_py38_parsers.py
    python tests/test_py38_parsers.py --verbose
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TASKS = REPO / "tasks"
PKG_CONFIGS = REPO / "vendor" / "pkg_configs"

# Base images whose default interpreter is Python 3.8.
PY38_MARKERS = ("ubuntu20.04", "-20.04", "22.04-py3")

FUTURE = "from __future__ import annotations"


def base_image(pkg: str) -> str:
    cfg = PKG_CONFIGS / pkg / "config.json"
    if not cfg.exists():
        return ""
    try:
        return json.loads(cfg.read_text()).get("base_image", "") or ""
    except json.JSONDecodeError:
        return ""


def py38_packages(task_dir: Path) -> list[tuple[str, str]]:
    cfg = task_dir / "config.json"
    if not cfg.exists():
        return []
    try:
        conf = json.loads(cfg.read_text())
    except json.JSONDecodeError:
        return []
    pkgs = {e.get("package") for e in (conf.get("test_cmds") or []) if e.get("package")}
    out = []
    for p in sorted(pkgs):
        img = base_image(p)
        if any(m in img for m in PY38_MARKERS):
            out.append((p, img))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    checked, failures = [], []
    for task_dir in sorted(TASKS.iterdir()):
        if not task_dir.is_dir() or task_dir.name == "deprecated":
            continue
        parser = task_dir / "parser.py"
        if not parser.exists():
            continue
        pkgs = py38_packages(task_dir)
        if not pkgs:
            continue
        has_future = FUTURE in parser.read_text()
        checked.append((task_dir.name, pkgs[0], has_future))
        if not has_future:
            failures.append((task_dir.name, pkgs[0]))

    if a.verbose:
        for name, (pkg, img), ok in checked:
            print(f"  {'ok  ' if ok else 'FAIL'} {name:34s} {pkg:16s} {img}")

    print(f"{len(checked) - len(failures)}/{len(checked)} py3.8 task parsers defer annotations")
    if failures:
        print(f"\n{len(failures)} parser(s) missing '{FUTURE}':")
        for name, (pkg, img) in failures:
            print(f"  - tasks/{name}/parser.py   ({pkg}, {img})")
        print("\nAdd the import right after the module docstring.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
