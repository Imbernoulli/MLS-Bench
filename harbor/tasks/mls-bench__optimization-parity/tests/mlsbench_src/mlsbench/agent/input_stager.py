"""Host-side re-stager for pre-generated task input blobs.

Some tasks withhold data from the agent's editable code (held-out targets,
hidden training labels, benchmark lookup tables). Their ``edits/mid_edit.py``
materializes that data into the workspace as opaque create-op blobs, and the
FIXED runner loads the blobs into memory and then DELETES them from disk
before any agent-editable hook executes (gated on MLSBENCH_EPHEMERAL_INPUTS=1,
which those tasks' eval scripts export). Editable code that later globs the
workspace therefore finds nothing.

Because each evaluation consumes (deletes) its inputs, they must be re-staged
before every test command. Such tasks opt in with ``"ephemeral_inputs": true``
in ``tasks/<t>/config.json``; the harness then runs this module host-side
right before each (test_cmd, seed) execution — host-side because mid_edit
imports the task's secret-bearing generator from ``holdout/<task>/`` (present
only in holdout-bearing checkouts, exactly like workspace setup; it is never
bind-mounted into the eval container). This mirrors the Harbor verifier's
``tests/eval/_inputgen/apply.py``, which re-materializes inputs before every
Harbor evaluation.

With ENV (test label) and SEED exported, the opted-in tasks' mid_edits
materialize ONLY the active run's blobs, so re-staging is cheap; a content
cache under ``<workspace_task_dir>/.input_cache`` (never bind-mounted into
any container) makes repeated re-staging a hardlink/copy instead of a
regeneration.

Usage:
    python -m mlsbench.agent.input_stager <task_name> <tasks_dir> <workspace_task_dir>

Environment:
    ENV / SEED               selective materialization (read by mid_edit)
    MLSBENCH_INPUT_CACHE=0   disable the host-side content cache
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path


def _normalize_pkg_name(name: str) -> str:
    """Mirror WorkspaceTools._normalize_pkg_name."""
    return name.lower().replace("-", "").replace("_", "")


def _resolve_dst(workspace_task_dir: Path, filename: str) -> Path:
    """Resolve an op filename to a workspace path (package-dir aware).

    Mirrors WorkspaceTools._resolve_workspace_path's package-relative branch:
    the first path component names a workspace package dir (normalized
    matching); fall back to a literal join under the workspace task dir.
    """
    parts = filename.split("/")
    if len(parts) > 1 and workspace_task_dir.is_dir():
        norm = _normalize_pkg_name(parts[0])
        for child in workspace_task_dir.iterdir():
            if child.is_dir() and _normalize_pkg_name(child.name) == norm:
                return child.joinpath(*parts[1:])
    return workspace_task_dir / filename


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _atomic_link_or_copy(src: Path, dst: Path) -> None:
    """Materialize dst with src's content atomically (hardlink, else copy)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / f".{dst.name}.{os.getpid()}.tmp"
    try:
        try:
            os.link(src, tmp)
        except OSError:
            shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _load_mid_edit_ops(mid_edit_file: Path, task_name: str) -> list[dict]:
    """Exec the task's mid_edit under a unique module name and return OPS.

    ENV / SEED from the current environment steer selective materialization
    (opted-in mid_edits stage only the active run's blobs when both are set).
    """
    spec = importlib.util.spec_from_file_location(
        "mlsbench_input_stager_mid_" + re.sub(r"[^A-Za-z0-9_]", "_", task_name),
        mid_edit_file,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(getattr(module, "OPS", []))


def _cache_key(task_dir: Path) -> str:
    """Content hash over everything the generated blobs depend on."""
    h = hashlib.sha256()
    for p in [task_dir / "edits" / "mid_edit.py", task_dir / "config.json"]:
        if p.is_file():
            h.update(p.name.encode())
            h.update(p.read_bytes())
    scripts = task_dir / "scripts"
    if scripts.is_dir():
        for p in sorted(scripts.glob("*.sh")):
            h.update(p.name.encode())
            h.update(p.read_bytes())
    env = re.sub(r"[^A-Za-z0-9._-]", "_", os.environ.get("ENV") or "ALL")
    seed = re.sub(r"[^A-Za-z0-9._-]", "_", os.environ.get("SEED") or "ALL")
    return f"{h.hexdigest()[:16]}_{env}_s{seed}"


def restage(task_name: str, tasks_dir: Path, workspace_task_dir: Path) -> int:
    """Re-materialize the active run's non-.py mid_edit create-ops.

    Never deletes or overwrites anything except the blob files themselves
    (atomic replace), and never touches .py files — the agent's editable
    program is left alone.
    """
    task_dir = tasks_dir / task_name
    mid_edit_file = task_dir / "edits" / "mid_edit.py"
    if not mid_edit_file.is_file():
        print(f"[input-stager] no mid_edit for {task_name}; nothing to do")
        return 0

    use_cache = os.environ.get("MLSBENCH_INPUT_CACHE", "1") != "0"
    cache_dir = workspace_task_dir / ".input_cache" / task_name / _cache_key(task_dir)
    marker = cache_dir / "COMPLETE.json"

    if use_cache and marker.is_file():
        try:
            names = json.loads(marker.read_text())["files"]
            if all((cache_dir / "files" / n).is_file() for n in names):
                for n in names:
                    _atomic_link_or_copy(
                        cache_dir / "files" / n, _resolve_dst(workspace_task_dir, n)
                    )
                print(
                    f"[input-stager] staged {len(names)} input file(s) for "
                    f"{task_name} (cache hit: {cache_dir.name})"
                )
                return 0
        except (json.JSONDecodeError, KeyError, OSError):
            pass  # fall through to regeneration

    ops = _load_mid_edit_ops(mid_edit_file, task_name)
    staged: list[str] = []
    for op in ops:
        if op.get("op") != "create":
            continue
        filename = op.get("file", "")
        if filename.endswith(".py"):
            continue  # never overwrite the agent's editable program
        content = op.get("content", "")
        if not content.endswith("\n"):
            content += "\n"  # mirror apply_pre_edit's normalization
        # The cache mirrors the op's full relative path under files/; the
        # marker records the same paths for workspace resolution on cache hits.
        if use_cache:
            cache_file = cache_dir / "files" / filename
            _atomic_write_text(cache_file, content)
            _atomic_link_or_copy(cache_file, _resolve_dst(workspace_task_dir, filename))
        else:
            _atomic_write_text(_resolve_dst(workspace_task_dir, filename), content)
        staged.append(filename)
    if use_cache:
        _atomic_write_text(marker, json.dumps({"files": staged}, indent=1))
    print(
        f"[input-stager] staged {len(staged)} input file(s) for {task_name} "
        f"(generated{'; cached as ' + cache_dir.name if use_cache else ''})"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 3:
        print(
            "usage: python -m mlsbench.agent.input_stager "
            "<task_name> <tasks_dir> <workspace_task_dir>",
            file=sys.stderr,
        )
        return 2
    return restage(argv[0], Path(argv[1]).resolve(), Path(argv[2]).resolve())


if __name__ == "__main__":
    sys.exit(main())
