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
    python -m mlsbench.agent.input_stager <task_name> <tasks_dir> <workspace_task_dir> \
        [--list-out FILE]

``--list-out`` records the workspace paths this stager installs, one per
line, INCREMENTALLY: each destination (and its temp file) is appended and
flushed BEFORE any of its bytes reach the workspace, so even after a mid-run
crash or SIGKILL the list is the authoritative superset of everything that
was (partially) installed. The harness uses it as the exact scope for its
host-side backstop scrub — never a broad glob. On an internal failure the
stager also unlinks everything it installed so far before exiting non-zero
(belt and braces); the caller must still treat a non-zero exit as fatal for
the entry and scrub whatever the list contains.

``--no-cache`` goes further than MLSBENCH_INPUT_CACHE=0: it also DELETES any
pre-existing ``.input_cache/<task>`` directory before staging. It exists as a
standalone/defensive CLI option for host-side runs of this stager: a
persistent cache keeps the withheld data readable even after the staged
destinations are scrubbed (including a cache left behind by an earlier
docker/apptainer run of the same workspace). NOTE the harness itself no
longer passes it — ``container_runtime: local`` is rejected outright for
ephemeral-input tasks (host-side agent code could read a later evaluation's
staged blobs; no cache policy can fix that). Container modes keep the cache:
it lives at ``<workspace_task_dir>/.input_cache``, a SIBLING of the
bind-mounted package dir, and the workspace_task_dir itself is never
bind-mounted.

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
import traceback
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


class _StagedList:
    """Incremental --list-out writer: one flushed line per recorded path."""

    def __init__(self, path: Path | None):
        self._fh = open(path, "w") if path is not None else None

    def add(self, p: Path) -> None:
        if self._fh is not None:
            self._fh.write(f"{p}\n")
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def _atomic_write_text(path: Path, text: str) -> None:
    """Atomic write for HOST-side cache/marker files (never agent-visible)."""
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


def _install_workspace_file(
    dst: Path,
    lw: _StagedList,
    *,
    src: Path | None = None,
    content: str | None = None,
) -> None:
    """Install one blob into the workspace, crash-accountably.

    The temp path AND the final path are appended (and flushed) to the
    --list-out record BEFORE any byte is written, so a SIGKILL at any point
    leaves the backstop scrub with the full scope. The final rename is atomic
    (concurrent readers never see a partial file)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / f".{dst.name}.{os.getpid()}.stage.tmp"
    lw.add(tmp)
    lw.add(dst)
    try:
        if src is not None:
            try:
                os.link(src, tmp)
            except OSError:
                shutil.copyfile(src, tmp)
        else:
            with open(tmp, "w") as fh:
                fh.write(content or "")
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
    """Content hash over everything the generated blobs depend on.

    Public side: mid_edit.py, config.json, scripts/*.sh. Private side: the
    holdout provider the generation imports — mid_edit resolves it as
    ``<project_root>/holdout/<task>/`` (project_root = tasks_dir.parent), and
    every file in that directory (dgp.py plus any data files it loads, e.g.
    nas's nb201_tables.json.gz) is folded into the key so a holdout change
    invalidates cached blobs even when no public file changed. When holdout/
    is absent (e.g. a public checkout with a self-contained mid_edit), the key
    carries an explicit ``noholdout`` token instead of a hash — a later
    holdout-bearing run therefore computes a different key and can never
    trust a holdout-blind cache entry (the marker records the same token).
    """
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
    holdout_dir = task_dir.parent.parent / "holdout" / task_dir.name
    if holdout_dir.is_dir():
        hh = hashlib.sha256()
        for p in sorted(holdout_dir.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(holdout_dir).as_posix()
            if "__pycache__" in rel or rel.endswith(".pyc"):
                continue
            hh.update(rel.encode())
            hh.update(p.read_bytes())
        holdout_tag = f"h{hh.hexdigest()[:12]}"
    else:
        holdout_tag = "noholdout"
    env = re.sub(r"[^A-Za-z0-9._-]", "_", os.environ.get("ENV") or "ALL")
    seed = re.sub(r"[^A-Za-z0-9._-]", "_", os.environ.get("SEED") or "ALL")
    return f"{h.hexdigest()[:16]}_{holdout_tag}_{env}_s{seed}"


def restage(
    task_name: str,
    tasks_dir: Path,
    workspace_task_dir: Path,
    list_out: Path | None = None,
    dry_run: bool = False,
    no_cache: bool = False,
) -> int:
    """Re-materialize the active run's non-.py mid_edit create-ops.

    Never deletes or overwrites anything except the blob files themselves
    (atomic replace), and never touches .py files — the agent's editable
    program is left alone. On an internal failure everything installed so far
    is unlinked before returning non-zero.

    ``dry_run=True`` installs NOTHING: it only records (via --list-out) the
    workspace destinations the active run's blobs resolve to — the harness
    uses this to derive the backstop-scrub scope for recovered jobs whose
    blobs were staged by a previous process.

    ``no_cache=True`` disables the persistent cache AND deletes any
    pre-existing ``.input_cache/<task>`` directory — for host-side contexts
    where a lingering cache would be readable outside the harness's control.
    """
    task_dir = tasks_dir / task_name
    mid_edit_file = task_dir / "edits" / "mid_edit.py"
    lw = _StagedList(list_out)
    installed: list[Path] = []
    try:
        if no_cache:
            stale_cache = workspace_task_dir / ".input_cache" / task_name
            if stale_cache.exists():
                shutil.rmtree(stale_cache, ignore_errors=True)
                print(
                    f"[input-stager] removed pre-existing cache for "
                    f"{task_name} (--no-cache: cache disabled)"
                )
        if not mid_edit_file.is_file():
            print(f"[input-stager] no mid_edit for {task_name}; nothing to do")
            return 0

        use_cache = (
            not no_cache
            and os.environ.get("MLSBENCH_INPUT_CACHE", "1") != "0"
        )
        cache_dir = (
            workspace_task_dir / ".input_cache" / task_name / _cache_key(task_dir)
        )
        marker = cache_dir / "COMPLETE.json"

        if use_cache and marker.is_file():
            try:
                names = json.loads(marker.read_text())["files"]
            except (json.JSONDecodeError, KeyError, OSError):
                names = None
            if names is not None and (
                dry_run
                or all((cache_dir / "files" / n).is_file() for n in names)
            ):
                for n in names:
                    dst = _resolve_dst(workspace_task_dir, n)
                    if dry_run:
                        lw.add(dst)
                        continue
                    _install_workspace_file(dst, lw, src=cache_dir / "files" / n)
                    installed.append(dst)
                print(
                    f"[input-stager] "
                    f"{'enumerated' if dry_run else 'staged'} {len(names)} "
                    f"input file(s) for {task_name} (cache hit: {cache_dir.name})"
                )
                return 0
            # fall through to regeneration

        ops = _load_mid_edit_ops(mid_edit_file, task_name)
        staged_names: list[str] = []
        for op in ops:
            if op.get("op") != "create":
                continue
            filename = op.get("file", "")
            if filename.endswith(".py"):
                continue  # never overwrite the agent's editable program
            content = op.get("content", "")
            if not content.endswith("\n"):
                content += "\n"  # mirror apply_pre_edit's normalization
            dst = _resolve_dst(workspace_task_dir, filename)
            if dry_run:
                lw.add(dst)
                staged_names.append(filename)
                continue
            if use_cache:
                # The cache mirrors the op's full relative path under files/;
                # cache writes are host-only (never agent-visible).
                cache_file = cache_dir / "files" / filename
                _atomic_write_text(cache_file, content)
                _install_workspace_file(dst, lw, src=cache_file)
            else:
                _install_workspace_file(dst, lw, content=content)
            installed.append(dst)
            staged_names.append(filename)
        if use_cache and not dry_run:
            _atomic_write_text(
                marker,
                json.dumps({"files": staged_names, "key": cache_dir.name}, indent=1),
            )
        print(
            f"[input-stager] "
            f"{'enumerated' if dry_run else 'staged'} {len(staged_names)} "
            f"input file(s) for {task_name} "
            f"({'dry run' if dry_run else 'generated' + ('; cached as ' + cache_dir.name if use_cache else '')})"
        )
        return 0
    except BaseException:
        # Belt and braces: unlink everything installed so far, then exit
        # non-zero. The caller must STILL backstop-scrub the --list-out
        # contents (covers SIGKILL, where this handler never runs).
        traceback.print_exc()
        for dst in installed:
            try:
                os.unlink(dst)
            except OSError:
                pass
        print(
            f"[input-stager] ERROR: staging failed for {task_name}; "
            f"removed {len(installed)} partially-staged file(s)",
            file=sys.stderr,
        )
        return 1
    finally:
        lw.close()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    list_out: Path | None = None
    if "--list-out" in argv:
        i = argv.index("--list-out")
        try:
            list_out = Path(argv[i + 1]).resolve()
        except IndexError:
            print("--list-out requires a path", file=sys.stderr)
            return 2
        del argv[i:i + 2]
    dry_run = False
    if "--dry-run" in argv:
        dry_run = True
        argv.remove("--dry-run")
    no_cache = False
    if "--no-cache" in argv:
        no_cache = True
        argv.remove("--no-cache")
    if len(argv) != 3:
        print(
            "usage: python -m mlsbench.agent.input_stager "
            "<task_name> <tasks_dir> <workspace_task_dir> "
            "[--list-out FILE] [--dry-run] [--no-cache]",
            file=sys.stderr,
        )
        return 2
    return restage(
        argv[0], Path(argv[1]).resolve(), Path(argv[2]).resolve(), list_out,
        dry_run, no_cache,
    )


if __name__ == "__main__":
    sys.exit(main())
