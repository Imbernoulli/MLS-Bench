#!/usr/bin/env python3
"""Harbor-side verifier for MLS-Bench tasks.

Runs three sub-commands inside the agent container:

    score_task.py guard       — edit-range diff guard
    score_task.py run-evals   — execute every setting's eval script
    score_task.py score       — aggregate metrics → combined_score → reward.txt

Designed to be self-contained — only stdlib, plus mlsbench installed in the
image (via the prebuilt mlsbench/<pkg> base image).
"""
from __future__ import annotations

import os
import sys

# test.sh runs every ROOT step as `PYTHONPATH=<site dirs> python -S` (-S keeps
# planted sitecustomize/.pth code out of root; PYTHONPATH keeps numpy/mlsbench
# importable). PYTHONPATH entries land AHEAD of the stdlib, so a pip backport
# in site-packages (dataclasses==0.6, typing) would shadow the stdlib here and
# crash the score step (ai4bio-protein-function-prediction /
# ai4sci-mol3d-generation: "module 'typing' has no attribute '_ClassVar'",
# reward 0). Put them back where site.py would: after the stdlib. Must run
# before any other import.
if sys.flags.no_site and os.environ.get("PYTHONPATH"):
    _pp = [p for p in os.environ["PYTHONPATH"].split(os.pathsep) if p]
    _moved = [p for p in sys.path if p in _pp]
    sys.path[:] = [p for p in sys.path if p not in _pp] + _moved
    del _pp, _moved

import argparse
import atexit
import hashlib
import json
import os
import re
import select
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import math
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path


# --------------------------------------------------------------------------- #
# Edit-range diff guard
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class EditRange:
    start: int  # 1-indexed inclusive; -1 means "whole file"
    end: int


def _load_task_config(task_meta: Path) -> dict:
    return json.loads((task_meta / "config.json").read_text())


def _editable_files(config: dict) -> dict[str, list[EditRange]]:
    out: dict[str, list[EditRange]] = {}
    for f in config.get("files", []):
        ranges = f.get("edit") or []
        filename = _safe_rel_path(str(f["filename"]))
        out[filename] = [EditRange(int(r["start"]), int(r["end"])) for r in ranges]
    return out


def _safe_rel_path(rel: str) -> str:
    if not rel or "\\" in rel:
        raise ValueError(f"unsafe workspace path: {rel!r}")
    p = Path(rel)
    if p.is_absolute() or any(part == ".." for part in p.parts):
        raise ValueError(f"unsafe workspace path: {rel!r}")
    return p.as_posix()


def _safe_join(root: Path, rel: str) -> Path:
    p = (root / _safe_rel_path(rel)).resolve()
    root_resolved = root.resolve()
    try:
        p.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"path escapes workspace root: {rel!r}") from exc
    return p


def _check_editable_only(
    pristine: Path,
    current: Path,
    ranges: list[EditRange],
) -> tuple[bool, str | None]:
    """Return (ok, reason) for whether `current` only differs from `pristine`
    inside the given editable ranges.

    The check is content-based, not line-number-based: it splits the pristine
    file into the alternating "fixed" / "editable" segments named by `ranges`,
    then verifies every fixed segment appears verbatim, in order, inside
    `current`. Whatever lies between the matched fixed segments is treated as
    the agent's edit; we don't care how long it is. This correctly handles
    replacements that change the line count (e.g. a 7-line baseline stub
    swapped for a 30-line implementation).

    If `pristine` doesn't exist, the agent created the file; the caller does
    not treat that as a violation.
    """
    if not pristine.exists():
        return False, "new file (no pristine)"

    pristine_text = pristine.read_text()
    current_text = current.read_text() if current.exists() else ""
    if pristine_text == current_text:
        return True, None
    if any(r.start == -1 and r.end == -1 for r in ranges):
        return True, None  # whole-file editable

    pristine_lines = pristine_text.splitlines(keepends=True)
    total_lines = len(pristine_lines)

    def _end_eff(r: EditRange) -> int:
        """`end=-1` means "to EOF" — normalize for indexing comparisons."""
        return total_lines if r.end == -1 else r.end

    # Build fixed segments from pristine.
    segments: list[list[str]] = []
    cursor = 0
    for r in sorted(ranges, key=lambda r: r.start):
        if r.start - 1 > cursor:
            segments.append(pristine_lines[cursor:r.start - 1])
        cursor = _end_eff(r)
    if cursor < total_lines:
        segments.append(pristine_lines[cursor:])

    # Match the FIRST segment at the start (if the file begins with a fixed
    # segment) and the LAST segment at the end (if the file ends with one).
    # Intermediate fixed segments are anchored at their rightmost feasible
    # occurrence between the surrounding anchors. A simple left-to-right greedy
    # match can grab duplicate text from an editable region and miss deletion of
    # the real fixed segment.
    starts_with_fixed = bool(segments) and (
        sorted(ranges, key=lambda r: r.start)[0].start > 1
    )
    ends_with_fixed = bool(segments) and (
        max(_end_eff(r) for r in ranges) < total_lines
    )

    fixed = ["".join(seg) for seg in segments]
    chosen: list[tuple[int, int] | None] = [None] * len(fixed)

    if starts_with_fixed and fixed:
        first = fixed[0]
        if first and not current_text.startswith(first):
            return False, (
                "submitted file does not start with the pristine's leading "
                "fixed segment — only the declared editable range may be modified"
            )
        chosen[0] = (0, len(first))

    if ends_with_fixed and fixed:
        last = fixed[-1]
        if last and not current_text.endswith(last):
            return False, (
                "submitted file does not end with the pristine's trailing "
                "fixed segment — only the declared editable range may be modified"
            )
        chosen[-1] = (len(current_text) - len(last), len(current_text))

    # Backward pass: for each segment, compute the latest occurrence that still
    # leaves room for every later fixed segment. This prevents an earlier copy
    # inside an editable range from stealing the anchor when the real fixed
    # segment is still present later.
    suffix: list[tuple[int, int] | None] = [None] * len(fixed)
    next_start = len(current_text)
    for i in range(len(fixed) - 1, -1, -1):
        seg = fixed[i]
        if chosen[i] is not None:
            suffix[i] = chosen[i]
            next_start = chosen[i][0]
            continue
        if not seg:
            suffix[i] = (next_start, next_start)
            continue
        idx = current_text.rfind(seg, 0, next_start)
        if idx < 0:
            return False, (
                f"fixed segment #{i + 1} not found in feasible order — only "
                "the declared editable range may be modified"
            )
        suffix[i] = (idx, idx + len(seg))
        next_start = idx

    prev_end = 0
    for i, seg in enumerate(fixed):
        if chosen[i] is not None:
            start, end = chosen[i]
        else:
            assert suffix[i] is not None
            start, end = suffix[i]
        if start < prev_end:
            return False, (
                f"fixed segment #{i + 1} overlaps an earlier fixed segment — "
                "only the declared editable range may be modified"
            )
        chosen[i] = (start, end)
        prev_end = end

    # With one editable interval, the anchored pristine prefix and suffix are
    # a complete proof that every fixed byte is unchanged.  SequenceMatcher
    # can misalign a repeated boundary line (commonly a blank line) with an
    # identical line inside a replacement, then falsely report the real fixed
    # boundary as deleted.  Multiple disjoint intervals still need the
    # line-level backstop below to reject deletion of an intermediate fixed
    # segment when the editable text contains a duplicate.
    if len(ranges) == 1:
        return True, None

    editable_line_nos = {
        line_no
        for r in ranges
        for line_no in (
            range(1, total_lines + 1)
            if r.start == -1
            else range(r.start, _end_eff(r) + 1)
        )
    }
    for tag, i1, i2, _j1, _j2 in SequenceMatcher(
        None, pristine_lines, current_text.splitlines(keepends=True), autojunk=False
    ).get_opcodes():
        if tag == "equal" or tag == "insert":
            continue
        changed_fixed = [
            str(line_no)
            for line_no in range(i1 + 1, i2 + 1)
            if line_no not in editable_line_nos
        ]
        if changed_fixed:
            return False, (
                "submitted file changes pristine fixed line(s) "
                f"{', '.join(changed_fixed[:5])} — only the declared editable "
                "range may be modified"
            )

    return True, None


_SKIP_DIR_PARTS = {".git", "__pycache__", "node_modules", ".pytest_cache", ".mypy_cache"}
_SKIP_SUFFIXES = {".pyc", ".pyo", ".so", ".o", ".egg-info"}
_EXEMPT_TOP_LEVEL_DIRS = {"_task"}


def _is_workspace_guard_exempt(rel: Path) -> bool:
    return bool(rel.parts and rel.parts[0] in _EXEMPT_TOP_LEVEL_DIRS)


def _is_path_excluded(rel: Path, excludes: tuple) -> bool:
    """True if *rel* (workdir-relative) is under a config ``guard_exclude`` prefix.

    Lets a task drop image-baked data dirs (datasets / weights / caches that the
    package install bakes INSIDE the package dir, e.g. ``dbim-codebase/assets``)
    out of the guard: they are not the agent's editable source surface, the
    render-time manifest never captured them, and walking them would otherwise
    flag every baked file as a spurious ``created``/``modified`` violation
    (issue #25.2 Error-1/Error-3). POSIX path-prefix match.
    """
    if not excludes:
        return False
    rp = rel.as_posix()
    return any(rp == e or rp.startswith(e + "/") for e in excludes)


def _walk_workspace(workspace_root: Path, excludes: tuple = ()) -> set[Path]:
    out: set[Path] = set()
    if not workspace_root.exists():
        return out
    for p in workspace_root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in _SKIP_DIR_PARTS for part in p.parts):
            continue
        if any(part.endswith(suf) for part in p.parts for suf in _SKIP_SUFFIXES):
            continue
        rel = p.relative_to(workspace_root)
        if _is_workspace_guard_exempt(rel):
            continue
        if _is_path_excluded(rel, excludes):
            continue
        out.add(rel)
    return out



# --------------------------------------------------------------------------- #
# Module shadowing anywhere in the workspace
#
# Trusted root launchers of keep-root tasks run `python -`, `python -c` or
# `python <script>` with cwd=/workspace or next to their own script, and
# sys.path[0] is then the cwd / the script's directory. A file the agent
# planted there -- /workspace/hashlib.py, a sourceless tempfile.pyc, an
# extension module json.cpython-311-x86_64-linux-gnu.so, a hashlib/ package --
# is imported INSTEAD of the stdlib module, as root, inside the trusted guard.
# The manifest check above only covers guarded package prefixes and skips
# .so/.pyc entirely. So, for EVERY task (allow_create does not license this):
#   * a new file anywhere under the workspace whose import name is a stdlib,
#     builtin or installed top-level module is a violation;
#   * under a guarded prefix with allow_create false, a new .so/.pyc/.pyo is a
#     violation too (the walk above never saw them);
#   * every __pycache__ in the workspace is removed before the evals: a planted
#     cpython-*.pyc whose header matches a trusted module's source would be
#     loaded in place of that source.
# --------------------------------------------------------------------------- #

def _import_name(rel: Path):
    """The top-level import name a planted file would answer to, or None."""
    name = rel.name
    if "__pycache__" in rel.parts:
        return None
    if name == "__init__.py" and len(rel.parts) >= 2:
        return rel.parts[-2]
    for suf in (".py", ".pyc", ".pyo", ".pyw"):
        if name.endswith(suf):
            return name[: -len(suf)]
    if name.endswith((".so", ".pyd")):
        return name.split(".", 1)[0]
    return None


def _shadowable_names(workspace_root: Path) -> set:
    import sysconfig
    names = set(sys.builtin_module_names)
    names |= set(getattr(sys, "stdlib_module_names", ()))
    roots = []
    for key in ("stdlib", "platstdlib", "purelib", "platlib"):
        p = sysconfig.get_path(key)
        if p:
            roots.append(p)
    roots += [p for p in sys.path if p]
    ws = str(workspace_root.resolve())
    for r in roots:
        try:
            r = os.path.realpath(r)
        except OSError:
            continue
        if r.startswith(ws) or r.startswith(("/tmp", "/tests", "/logs")) or not os.path.isdir(r):
            continue
        for d in (r, os.path.join(r, "lib-dynload")):
            try:
                entries = os.listdir(d)
            except OSError:
                continue
            for e in entries:
                if e.startswith((".", "_distutils")) or e.endswith((".dist-info", ".egg-info", ".pth")):
                    continue
                n = _import_name(Path(e)) if "." in e else e
                if n:
                    names.add(n)
    names.discard("__init__")
    return names


def _purge_workspace_bytecode(workspace_root: Path) -> int:
    n = 0
    for dirpath, dirnames, _files in os.walk(str(workspace_root)):
        if "__pycache__" in dirnames:
            shutil.rmtree(os.path.join(dirpath, "__pycache__"), ignore_errors=True)
            dirnames.remove("__pycache__")
            n += 1
        dirnames[:] = [d for d in dirnames if not os.path.islink(os.path.join(dirpath, d))]
    return n


def cmd_guard(args: argparse.Namespace) -> int:
    task_meta = Path(args.task_meta)
    config = _load_task_config(task_meta)
    # config.json::files[].filename is relative to the workdir (e.g.
    # "causal-learn/bench/custom_algorithm.py"). The pristine root holds
    # declared-file content for byte-segment matching; the manifest holds
    # sha256 for every file under any guarded prefix. Both are uploaded
    # fresh by Harbor at verify time so the agent cannot tamper with them.
    pristine_root = Path(args.pristine)
    workspace_root = Path(args.workspace)
    violation_out = Path(args.violation_out)

    manifest_path = task_meta / "pristine_manifest.json"
    if not manifest_path.exists():
        # Fail closed: missing manifest is an adapter packaging bug, but
        # silently treating it as "no constraints" lets the agent edit any
        # non-declared file. Refuse to grade.
        violation_out.parent.mkdir(parents=True, exist_ok=True)
        violation_out.write_text(
            "pristine_manifest.json missing — refusing to verify\n"
        )
        return 10
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        violation_out.parent.mkdir(parents=True, exist_ok=True)
        violation_out.write_text("pristine_manifest.json malformed\n")
        return 10
    if not isinstance(manifest, dict) or not manifest:
        violation_out.parent.mkdir(parents=True, exist_ok=True)
        violation_out.write_text(
            "pristine_manifest.json empty — refusing to verify\n"
        )
        return 10

    violations: list[str] = []

    editable = _editable_files(config)
    # Optional per-task allowlist of workdir-relative prefixes dropped from the
    # guard — for image-baked data/build dirs inside the package dir that aren't
    # the agent's editable surface and were never in the render-time manifest
    # (issue #25.2 Error-1/Error-3).
    guard_exclude = tuple(config.get("guard_exclude", []) or [])
    # `allow_create: false` (the default) means the instruction told the agent
    # that creating a new file under a guarded prefix scores zero — so enforce
    # it. `allow_create: true` (8 tasks) leaves creation unrestricted.
    allow_create = bool(config.get("allow_create", False))

    workspace_files = _walk_workspace(workspace_root, guard_exclude)
    workspace_rel_strs = {p.as_posix() for p in workspace_files}

    # Guarded prefixes: every top-level dir referenced by editable list AND
    # the manifest (covers secondary packages even if no declared edits).
    guarded_prefixes = {Path(f).parts[0] for f in editable if f}
    guarded_prefixes |= {Path(f).parts[0] for f in manifest if f}

    # Created files. The guard never DELETES them (deletion between guard and
    # eval silently broke a legitimate helper-module split, dying on ImportError
    # with nothing to show why). But whether creating one is a *violation* now
    # follows `allow_create`:
    #
    #   * `allow_create: false` (the default, 1262 tasks): the instruction told
    #     the agent that creating a new file scores zero. A file that appears
    #     under a guarded prefix, is not in the render-time manifest, is not a
    #     declared editable file, and is not under a guard_exclude prefix is
    #     therefore a violation. This closes the module-shadowing threat some
    #     tasks' designs rely on (a planted `numpy.py` beside a trusted
    #     launcher) at the guard level rather than leaning on the launcher's
    #     own `python -I`. Scoped to guarded prefixes because that is the
    #     agent's editable surface and the one region with a clean manifest
    #     baseline; a brand-new top-level scratch dir the eval never imports is
    #     left alone (and cannot reach the sealed verifier anyway).
    #   * `allow_create: true` (8 tasks): creation is unrestricted, exactly as
    #     before — a helper module the editable file imports is fine.
    #
    # Absence from the manifest never means "the agent made this" for baked data
    # (dbim-codebase/assets, badge/oml): those are dropped via guard_exclude, so
    # the check below only sees them when a task forgot to exclude them — the
    # same hand-maintained allowlist the deletion path already depends on.

    # Disallowed deletion: anything in manifest under a guarded prefix that
    # is gone from workspace.
    for rel_str in sorted(manifest):
        rel = Path(rel_str)
        if not rel.parts or rel.parts[0] not in guarded_prefixes:
            continue
        if _is_path_excluded(rel, guard_exclude):
            continue
        if rel_str in workspace_rel_strs:
            continue
        if rel_str in editable:
            # Declared editable files: treat deletion as a range violation
            # so the existing logic below produces a more specific message.
            continue
        violations.append(f"deleted file: {rel_str}")

    # Edit-range checks for files declared with allowed-edit ranges.
    for rel_name, ranges in editable.items():
        cur = _safe_join(workspace_root, rel_name)
        pri = _safe_join(pristine_root, rel_name)
        if not pri.exists():
            # Adapter packaging bug — every declared file should have a
            # pristine. Whole-file editable still requires the file to
            # exist in workspace; missing pristine cannot excuse deletion.
            if any(r.start == -1 and r.end == -1 for r in ranges):
                if not cur.exists():
                    violations.append(
                        f"{rel_name}: deleted (whole-file editable but missing in workspace)"
                    )
                continue
            violations.append(f"{rel_name}: pristine snapshot missing in tests/meta/pristine")
            continue
        if not ranges:
            # Declared read-only.
            if cur.exists() and cur.read_text() != pri.read_text():
                violations.append(f"{rel_name}: modified but file is declared read-only")
            elif not cur.exists():
                violations.append(f"{rel_name}: deleted but file is declared read-only")
            continue
        if not cur.exists():
            violations.append(f"{rel_name}: deleted (file declared with editable range)")
            continue
        ok, reason = _check_editable_only(pri, cur, ranges)
        if not ok:
            violations.append(f"{rel_name}: {reason}")

    # Modifications to files NOT declared in config.json::files[] but under a
    # guarded prefix: hash-compare against the manifest. Binary-safe.
    for rel in sorted(workspace_files):
        rel_str = rel.as_posix()
        if rel_str in editable:
            continue
        if not rel.parts or rel.parts[0] not in guarded_prefixes:
            continue
        expected_sha = manifest.get(rel_str)
        if expected_sha is None:
            # Newly created file under a guarded prefix. A violation unless the
            # task allows creation (see the block above).
            if not allow_create:
                violations.append(f"created file: {rel_str} (allow_create is false)")
            continue
        try:
            actual = hashlib.sha256(
                _safe_join(workspace_root, rel_str).read_bytes()
            ).hexdigest()
        except OSError:
            continue
        if actual != expected_sha:
            violations.append(f"{rel_str}: modified but not in editable file list")

    # Module shadowing and unguarded bytecode/extension files (see the block
    # above _import_name). Scans the WHOLE workspace, not only guarded prefixes.
    shadowable = _shadowable_names(workspace_root)
    shadow_scratch: list = []
    # Compiled artifacts (.so/.pyd/.pyc/.pyo) are NOT in the render-time
    # manifest: an image may compile an extension in-tree at build time. They
    # are reconciled against the BUILT image's own listing when the bundle
    # carries one (site_manifest.json "workspace_compiled", recorded post-build
    # by bake_interpreter_manifest.py) -- exact sha match or it is a created
    # file. Without that listing they are skipped under the guarded prefixes,
    # as the modification loop above skips them (a false zero for every arm is
    # worse than the narrow residual; docs/harbor.md).
    built_compiled = None
    _sm = task_meta / "site_manifest.json"
    if _sm.exists():
        try:
            built_compiled = json.loads(_sm.read_text()).get("workspace_compiled")
        except (OSError, json.JSONDecodeError):
            built_compiled = None
    for p in workspace_root.rglob("*"):
        try:
            if not (p.is_file() or (p.is_dir() and (p / "__init__.py").is_file())):
                continue
        except OSError:
            continue
        rel = p.relative_to(workspace_root)
        if _is_workspace_guard_exempt(rel) or _is_path_excluded(rel, guard_exclude):
            continue
        if any(part in (".git", "node_modules") for part in rel.parts):
            continue
        rel_str = rel.as_posix()
        if rel_str in manifest or rel_str in editable:
            continue
        compiled = p.is_file() and rel.name.endswith((".so", ".pyd", ".pyc", ".pyo")) \
            and "__pycache__" not in rel.parts
        in_tree = bool(rel.parts and rel.parts[0] in guarded_prefixes)
        if compiled and in_tree:
            if built_compiled is None:
                continue
            want_sha = built_compiled.get(rel_str)
            if want_sha is not None:
                try:
                    if hashlib.sha256(p.read_bytes()).hexdigest() == want_sha:
                        continue
                except OSError:
                    continue
                violations.append(f"{rel_str}: compiled file differs from the built image")
                continue
        if p.is_dir():
            if (rel_str + "/__init__.py") in manifest:
                continue
            mod = rel.name
        else:
            mod = _import_name(rel)
        if mod and mod in shadowable:
            if rel.parts and rel.parts[0] in guarded_prefixes:
                # Inside the task's own source tree: a submission change.
                violations.append(
                    f"created file shadows the module '{mod}': {rel_str} (a trusted "
                    "launcher importing from this directory would load it instead)"
                )
            else:
                # Scratch space outside the source trees is "not part of your
                # submission" (instruction.md): an agent's /workspace/test.py
                # is common and honest, so it is removed, not scored zero.
                shadow_scratch.append(p / "__init__.py" if p.is_dir() else p)
            continue
        if (not allow_create and p.is_file() and rel.parts and rel.parts[0] in guarded_prefixes
                and "__pycache__" not in rel.parts
                and rel.name.endswith((".so", ".pyd", ".pyc", ".pyo"))):
            violations.append(f"created file: {rel_str} (allow_create is false)")

    if violations:
        violation_out.parent.mkdir(parents=True, exist_ok=True)
        violation_out.write_text("\n".join(violations) + "\n")
        return 10
    removed = []
    for p in shadow_scratch:
        try:
            p.unlink()
            removed.append(str(p))
        except OSError:
            pass
    if removed:
        try:
            (violation_out.parent / "guard_shadow_removed.txt").write_text(
                "removed scratch files outside the source trees that would shadow a "
                "stdlib/installed module:\n" + "\n".join(removed) + "\n")
        except OSError:
            pass
    purged = _purge_workspace_bytecode(workspace_root)
    if purged:
        try:
            (violation_out.parent / "guard_bytecode_purge.txt").write_text(
                f"removed {purged} __pycache__ dir(s) from the workspace before the evals\n")
        except OSError:
            pass
    return 0


# --------------------------------------------------------------------------- #
# Run every setting's eval script
# --------------------------------------------------------------------------- #

_ENV_VAR_RE = re.compile(r"\$(\w+)|\$\{([^}:]+)(?::-[^}]*)?\}")


def _read_meta_text(task_meta: Path, name: str, default: str = "") -> str:
    p = task_meta / name
    if not p.exists():
        return default
    return p.read_text().strip() or default


def _expand_env_template(value: str, base_env: dict[str, str]) -> str:
    def repl(match):
        name = match.group(1) or match.group(2) or ""
        return base_env.get(name, "")

    return _ENV_VAR_RE.sub(repl, value)


def _load_package_envs(task_meta: Path) -> dict[str, dict[str, str]]:
    p = task_meta / "package_envs.json"
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}
    return {
        str(pkg): {str(k): str(v) for k, v in env.items()}
        for pkg, env in raw.items()
        if isinstance(env, dict)
    }


def _package_dir(workspace_root: Path, default_pkg: str, tc: dict) -> Path:
    pkg = str(tc.get("package") or default_pkg)
    candidate = workspace_root / pkg
    if candidate.exists():
        return candidate
    norm = _normalize_pkg_name(pkg)
    if workspace_root.exists():
        for child in workspace_root.iterdir():
            if child.is_dir() and _normalize_pkg_name(child.name) == norm:
                return child
    return workspace_root / default_pkg


def _normalize_pkg_name(name: str) -> str:
    return str(name).lower().replace("-", "").replace("_", "")


def _eval_env(
    *,
    task_meta: Path,
    out_dir: Path,
    workspace_root: Path,
    pkg_dir: Path,
    tc: dict,
    seed: int,
    unprivileged: tuple[int, int] | None = None,
    task_dir: Path | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    # test.sh starts this ROOT process as `PYTHONPATH=<site dirs> python -S`
    # only so that -S still finds numpy/mlsbench; that value must not reach
    # the eval children. Left in, a package env "X:${PYTHONPATH}" expands it
    # and puts site-packages AHEAD of the stdlib in every eval / budget_check
    # process, where a pip backport (dataclasses==0.6, typing) then shadows
    # the stdlib and crashes `import torch`. test.sh unsets the agent's
    # PYTHONPATH on entry, so dropping it here restores the children's default
    # environment (no PYTHONPATH unless the package env sets one, with
    # ${PYTHONPATH} expanding to empty).
    env.pop("PYTHONPATH", None)
    default_pkg = _read_meta_text(task_meta, "package", "")
    package_envs = _load_package_envs(task_meta)
    pkg_name = str(tc.get("package") or default_pkg)
    for key, value in package_envs.get(pkg_name, package_envs.get(default_pkg, {})).items():
        if key == "HOME":
            env[key] = value
        else:
            env[key] = _expand_env_template(value, env)

    # Command-specific environment overrides (including the materialized
    # H200 block) must reach the actual eval subprocess.  Keep this after the
    # package defaults so a task command can intentionally specialize them,
    # and expand templates against the already-populated environment.
    for key, value in (tc.get("env") or {}).items():
        key = str(key)
        value = str(value)
        env[key] = _expand_env_template(value, env)

    task_id = _read_meta_text(task_meta, "task_id", "unknown")
    save_path = out_dir / "save"
    output_dir = save_path / task_id / "harbor" / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    env["SAVE_PATH"] = str(save_path)
    env["OUTPUT_DIR"] = str(output_dir)
    env["SEED"] = str(seed)
    label = str(tc.get("label", ""))
    if label:
        env["ENV"] = label
    # Under the privilege drop `task_dir` is the throwaway eval-time copy
    # (_build_eval_task_dir) that /workspace/_task also points at: the real
    # task_meta is sealed 0700, and it holds parser.py / score_spec.py /
    # leaderboard.csv / dgp.py, none of which eval-time code may read.
    env["MLSBENCH_TASK_DIR"] = str(task_dir if task_dir is not None else task_meta)
    env["MLSBENCH_PKG_DIR"] = str(pkg_dir)
    # The shared hardened launcher for a task's TRUSTED root python (see
    # trusted_run). /tests/score_task.py is the verifier's own read-only copy.
    _vpy = os.environ.get("MLSBENCH_VERIFIER_PYTHON") or sys.executable
    env["MLSBENCH_TRUSTED_PYTHON"] = f"{_vpy} -I -S -B /tests/score_task.py trusted-run --"
    # The shared helper that runs a keep-root task's UNTRUSTED phase under a
    # fresh uid and leaves nothing of it behind (cmd_run_untrusted).
    env["MLSBENCH_RUN_UNTRUSTED"] = f"{_vpy} -I -S -B /tests/score_task.py run-untrusted"
    env.setdefault("DATA_ROOT", "/data")
    env["MLSBENCH_LOCAL_PATH_MAP_JSON"] = json.dumps({
        "/workspace": str(workspace_root),
        f"/workspace/{pkg_name}": str(pkg_dir),
        "/data": "/data",
    })
    if unprivileged is not None:
        _apply_unprivileged_env(env, *unprivileged)
        # SAVE_PATH / OUTPUT_DIR were just created by this (root) process.
        for path in (save_path, output_dir):
            try:
                os.chmod(path, 0o777)
            except OSError:
                pass
    return env


def _parse_time_to_seconds(time_str: str) -> int:
    parts = str(time_str).split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        return int(float(parts[0]))
    except (ValueError, IndexError):
        return 3600


def _test_cmd_compute(tc: dict) -> float:
    try:
        return float(tc.get("compute", 1) or 1)
    except (TypeError, ValueError):
        return 1.0


def _running_gpu_is_h200() -> bool:
    """Return whether this verifier is running on an H200-class GPU.

    MLS-Bench's native runner selects the H200-specific command block only
    through an explicit host setting (``compute_scale: 0.5``); it never
    auto-detects the device.  Harbor mirrors that: the provider must export
    ``MLSBENCH_GPU_TYPE=H200`` (for example via ``--ek gpu_type=H200``).
    Running the baseline command on an H200 host without that variable keeps
    the H100 profile, exactly like native runs.
    """
    raw = os.environ.get("MLSBENCH_GPU_TYPE", "").strip()
    return "H200" in raw.upper()


def _effective_test_cmds(config: dict) -> list[dict]:
    """Materialize hardware-specific test command overrides.

    The ``h200`` blocks are deliberately metadata in the public task config:
    they are selected only when an H200 is really attached.  Keep the original
    dictionaries untouched so the verifier's budget and scheduling code sees
    one consistent, fully materialized command list.
    """
    entries = [dict(tc) for tc in (config.get("test_cmds", []) or [])]
    if not _running_gpu_is_h200():
        return entries
    for entry in entries:
        override = entry.pop("h200", None)
        if not isinstance(override, dict):
            continue
        if "cmd" in override:
            entry["cmd"] = override["cmd"]
        if "compute" in override:
            entry["compute"] = override["compute"]
        if override.get("env"):
            env = dict(entry.get("env") or {})
            env.update(override["env"])
            entry["env"] = env
    return entries


def _config_seeds(config: dict) -> list[int]:
    seeds = config.get("seeds") or [42]
    if isinstance(seeds, int):
        seeds = [seeds]
    return sorted(int(seed) for seed in seeds)


def _group_entries(test_cmds: list[dict]) -> dict[int, list[tuple[int, dict]]]:
    auto_group = 10000
    grouped: dict[int, list[tuple[int, dict]]] = {}
    for idx, entry in enumerate(test_cmds):
        group = entry.get("group")
        if group is None:
            group = auto_group
            auto_group += 1
        grouped.setdefault(group, []).append((idx, entry))
    return grouped


# Peak GPUs a task may reserve; groups that want more at once are run as
# sequential waves of at most this many GPUs. Only consulted when
# tests/meta/gpu_count is missing — the rendered value is authoritative. Keep in
# sync with `MAX_PARALLEL_GPUS` in harbor_adapter/src/mls_bench/adapter.py.
MAX_PARALLEL_GPUS = 8

# Grace added to every wave's deadline. adapter.py::_verifier_timeout_sec
# charges the same amount per wave, so the outer verifier budget can never be
# tighter than the deadlines this runner hands out.
WAVE_GRACE_SEC = 300
# Every wave's deadline is the declared `time` of its slowest command times
# this factor: the declared times were sized for the hardware the leaderboard
# was measured on, and a deadline that kills a correct run on a slower card
# scores it 0 with no other symptom. Mirrors adapter.py::EVAL_TIME_FACTOR.
EVAL_TIME_FACTOR = 3.0


def _bin_pack_fractional_gpus(fractionals: list[float]) -> int:
    """Min 1.0-capacity bins needed for these fractional GPU jobs.

    First-fit-decreasing. `ceil(sum)` underestimates whenever per-GPU capacity
    is bin-limited (e.g. 9 jobs of 0.4 GPUs each need 5 bins, not ceil(3.6)=4,
    because each bin holds at most floor(1/0.4)=2 such jobs).
    """
    bins: list[float] = []
    for c in sorted(fractionals, reverse=True):
        for i, cap in enumerate(bins):
            if cap >= c:
                bins[i] = cap - c
                break
        else:
            bins.append(1.0 - c)
    return len(bins)


def _peak_gpu_usage(computes: list[float]) -> int:
    """GPUs needed to run every job in *computes* at once."""
    whole = sum(max(1, math.ceil(c)) for c in computes if c >= 1.0)
    return whole + _bin_pack_fractional_gpus([c for c in computes if 0.0 < c < 1.0])


def _split_computes_into_waves(computes: list[float], devices: int) -> list[list[float]]:
    """Greedy contiguous batching, same order as _partition_group_gpu_batches."""
    waves: list[list[float]] = []
    current: list[float] = []
    for compute in computes:
        if current and _peak_gpu_usage([*current, compute]) > devices:
            waves.append(current)
            current = [compute]
        else:
            current.append(compute)
    if current:
        waves.append(current)
    return waves


def _wave_balanced_gpus(
    computes: list[float],
    peak_gpus: int,
    largest_job: int,
) -> int:
    """Widest reservation <= MAX_PARALLEL_GPUS whose waves are all equally full.

    Mirrors adapter.py::_wave_balanced_gpus; see the rationale there.
    """
    if peak_gpus <= MAX_PARALLEL_GPUS:
        return peak_gpus
    for candidate in range(MAX_PARALLEL_GPUS, largest_job - 1, -1):
        usage = [
            _peak_gpu_usage(wave) for wave in _split_computes_into_waves(computes, candidate)
        ]
        if usage and len(set(usage)) == 1:
            return usage[0]
    return largest_job


def _infer_reserved_gpu_count(config: dict) -> int:
    if config.get("use_cuda") is False:
        return 0
    test_cmds = list(config.get("test_cmds", []) or [])
    if not config.get("use_cuda") and not any("compute" in tc for tc in test_cmds):
        return 0

    peak_gpus = 0
    largest_job = 0
    peak_computes: list[float] = []
    n_seeds = max(1, len(_config_seeds(config)))
    for entries in _group_entries(test_cmds).values():
        whole = 0
        fractionals: list[float] = []
        computes: list[float] = []
        for _, tc in entries:
            compute = _test_cmd_compute(tc)
            computes.extend([compute] * n_seeds)
            if compute > 0.0:
                largest_job = max(largest_job, max(1, math.ceil(compute)))
            if compute >= 1.0:
                whole += n_seeds * max(1, math.ceil(compute))
            elif compute > 0.0:
                fractionals.extend([compute] * n_seeds)
        group_peak = whole + _bin_pack_fractional_gpus(fractionals)
        if group_peak >= peak_gpus:
            peak_computes = computes
        peak_gpus = max(peak_gpus, group_peak)
    if not peak_gpus:
        return 0
    # Same rule the adapter applies when rendering tests/meta/gpu_count: a group
    # wanting more GPUs at once than MAX_PARALLEL_GPUS runs as sequential waves
    # (see _partition_group_gpu_batches), and the reservation is the widest one
    # at or below the cap that every wave fills. Clamping straight to the cap
    # would leave the final wave holding idle GPUs — three 4-GPU jobs on 8
    # reserved run as waves of 2 and 1, costing more GPU-hours than reserving 4
    # and more wall clock than reserving 12.
    return max(1, _wave_balanced_gpus(peak_computes, peak_gpus, largest_job), largest_job)


def _reserved_gpu_count(task_meta: Path, config: dict) -> int:
    p = task_meta / "gpu_count"
    if p.exists():
        try:
            return max(0, int(p.read_text().strip() or "0"))
        except ValueError:
            pass
    return _infer_reserved_gpu_count(config)


def _visible_gpu_indices(task_meta: Path, config: dict) -> list[str]:
    reserved = _reserved_gpu_count(task_meta, config)
    if reserved <= 0:
        return []

    # Source-of-truth ordering inside the container:
    #   1. nvidia-smi — reflects what nvidia-container-runtime actually
    #      attached via docker-compose `deploy.resources.reservations`.
    #      Authoritative.
    #   2. CUDA_VISIBLE_DEVICES env var — many base images (pytorch/conda)
    #      pre-set this to "0" as a single-GPU default. Trusting that
    #      env var would silently cap our scheduler at 1 GPU even when
    #      docker actually exposes more. Only use as a fallback when
    #      nvidia-smi is unavailable.
    #   3. range(reserved) — last-resort fallback.
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        devices = [line.strip() for line in out.stdout.splitlines() if line.strip()]
        if devices:
            return devices[:reserved]
    except Exception:
        pass

    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if raw and raw.lower() not in {"all", "none", "void", "-1"}:
        devices = [d.strip() for d in raw.split(",") if d.strip()]
        if devices:
            return devices[:reserved]

    return [str(i) for i in range(reserved)]


def _task_gpu_need(task: dict) -> int:
    compute = _test_cmd_compute(task["entry"]["tc"])
    if compute <= 0.0:
        return 0
    if compute >= 1.0:
        return max(1, math.ceil(compute))
    return 1


def _try_allocate_task_to_remaining(
    task: dict,
    remaining: dict[str, float],
) -> str | None:
    compute = _test_cmd_compute(task["entry"]["tc"])
    if compute <= 0.0:
        return None
    if compute >= 1.0:
        need = max(1, math.ceil(compute))
        free = [device for device, cap in remaining.items() if cap >= 1.0]
        if len(free) < need:
            return None
        chosen = free[:need]
        for device in chosen:
            remaining[device] = 0.0
        return ",".join(chosen)

    chosen = next((device for device, cap in remaining.items() if cap >= compute), None)
    if chosen is None:
        return None
    remaining[chosen] -= compute
    return chosen


def _allocate_group_gpu_assignments(
    tasks: list[dict],
    devices: list[str],
) -> list[str | None] | None:
    if not devices:
        return [None] * len(tasks)

    assignments: list[str | None] = [None] * len(tasks)
    remaining = {device: 1.0 for device in devices}
    indexed = list(enumerate(tasks))
    indexed.sort(
        key=lambda item: (
            0 if _test_cmd_compute(item[1]["entry"]["tc"]) >= 1.0 else 1
        )
    )

    for idx, task in indexed:
        assignment = _try_allocate_task_to_remaining(task, remaining)
        if assignment is None and _task_gpu_need(task) > 0:
            return None
        assignments[idx] = assignment
    return assignments


def _partition_group_gpu_batches(
    tasks: list[dict],
    devices: list[str],
) -> list[tuple[list[dict], list[str | None]]] | None:
    if not devices:
        return [(list(tasks), [None] * len(tasks))]

    batches: list[tuple[list[dict], list[str | None]]] = []
    current: list[dict] = []
    for task in tasks:
        trial = [*current, task]
        if _allocate_group_gpu_assignments(trial, devices) is None:
            if not current:
                return None
            assignments = _allocate_group_gpu_assignments(current, devices)
            if assignments is None:
                return None
            batches.append((current, assignments))
            current = [task]
            if _allocate_group_gpu_assignments(current, devices) is None:
                return None
        else:
            current = trial

    if current:
        assignments = _allocate_group_gpu_assignments(current, devices)
        if assignments is None:
            return None
        batches.append((current, assignments))
    return batches


def _process_group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except PermissionError:
        return True
    except OSError:
        return False


def _kill_process_group(pgid: int, timeout: float = 30.0) -> None:
    try:
        os.killpg(pgid, signal.SIGTERM)
    except OSError:
        return

    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _process_group_alive(pgid):
            return
        time.sleep(0.5)

    try:
        os.killpg(pgid, signal.SIGKILL)
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# Trusted capture of the eval command's stdout
#
# The scored metric is a line the eval script prints; the parser reads it back
# from this log. So WHO can write this log is a trust boundary, and until now it
# was the wrong side of one. `subprocess.Popen(stdout=fh)` handed the eval's fd 1
# to an OPEN FILE, which the eval command and every process it spawns INHERIT as
# a writable descriptor. `start_new_session=True` + `_kill_process_group` tear
# down the session the verifier created, but a submission that calls `setsid()`
# (or double-forks) escapes that session, OUTLIVES the eval script, and then —
# still holding the inherited fd — appends a second, forged marker after the
# honest one (parsers take the last match) or `lseek(0)`+rewrites the whole log.
# A survivor that reprints a forged `TEST_METRICS:` line can move a starter's
# reward up through the real verifier.
#
# The fix removes the inherited writable handle. The eval's fd 1 is now a PIPE
# whose read end only the verifier holds; the verifier copies the bytes into a
# root-owned capture file the eval can neither reopen (it runs unprivileged, or
# — in the opt-out mode — under the task's own uid drop) nor reach a descriptor
# to. And the verifier STOPS copying the instant the eval command exits: every
# honest byte, including the terminal marker the shell prints as its last act,
# is already in the pipe by then, while anything a survivor writes afterwards
# lands in a pipe nobody reads. `unshare --pid` would also reap survivors but
# needs CAP_SYS_ADMIN, which Docker/Daytona/Modal drop (measured: `unshare
# --pid` is `Operation not permitted` in the eval image), so this is the
# portable half of the same idea — provenance at the capture, not the kernel.
# --------------------------------------------------------------------------- #


def _drain_captured(state: dict) -> None:
    """Move whatever is buffered on the eval's stdout pipe into the capture file.

    Non-blocking, so it never waits on a process that has escaped the session and
    is holding the write end open with nothing more to say."""
    fd = state.get("read_fd")
    cap = state.get("cap_fh")
    if fd is None or cap is None:
        return
    while True:
        try:
            chunk = os.read(fd, 1 << 16)
        except BlockingIOError:
            return
        except OSError:
            return
        if not chunk:  # EOF: every write end (incl. any survivor) has closed
            return
        cap.write(chunk)


def _stop_capture(state: dict) -> None:
    """Freeze the log at the eval command's exit.

    One final non-blocking drain grabs the honest bytes already in the pipe (the
    shell writes its authoritative marker, then exits), then the read end is
    closed so a survivor's later writes get EPIPE and are never seen, and the
    capture file is closed. The scorer only ever reads bytes captured before this
    point."""
    _drain_captured(state)
    fd = state.pop("read_fd", None)
    if fd is not None:
        try:
            os.close(fd)
        except OSError:
            pass
    cap = state.get("cap_fh")
    if cap is not None:
        try:
            cap.flush()
            cap.close()
        except OSError:
            pass
        state["cap_fh"] = None


# --------------------------------------------------------------------------- #
# Survivor-process forgery and the trusted result channel
#
# The pipe above stops copying when the eval shell EXITS, but "exits" is only
# observed at the verifier's next poll. A survivor that fork()+setsid()s keeps
# the pipe's write end, spins on /proc/<shell>/stat until the shell is a
# zombie -- i.e. right after the honest terminal marker -- and writes a forged
# marker within ~0.2 ms, before the 100 ms poll noticed the exit; the final
# drain then captured it and a last-wins parser took it (rl-offline-adroit
# R2 review, 10/10 through the real _run_eval_wave). A nonce on stdout does not
# help: a same-uid attacker reads it from /proc/<pid>/environ or cmdline, and
# it can key on the honest writer's /proc/<pid>/io instead of the shell's exit.
# Bytes on a pipe carry no provenance, so while trusted and untrusted code
# share a uid and a stdout, NO capture-side rule can separate them.
#
# Two layers:
#   * defense in depth, every task: the exit is noticed on SIGCHLD (not the
#     next poll); at exit every process still holding a WRITE end of that
#     command's pipe and the command's own process group are SIGKILLed before
#     the final drain; at the end of each wave of a dropped eval, every
#     process of the eval uid is killed; `harbor_unique_markers` (config.json,
#     list of regexes) refuses a record whose parsed text does not contain
#     exactly one line matching each. The race SHRINKS (measured in
#     tests/test_harbor_survivor_forgery.py) but cannot close.
#   * the real fix, opt-in `"harbor_trusted_result": true`: the eval keeps
#     root; the task runs its untrusted phase through $MLSBENCH_RUN_UNTRUSTED
#     (cmd_run_untrusted: a fresh never-reused uid, its own stdout pipe, every
#     process of that uid SIGKILLed when the command exits, its files returned
#     to root, shared tmp + SysV IPC swept); its trusted phase (root) writes the
#     scored marker line(s) to $MLSBENCH_RESULT_FILE, a file in a root-only
#     0700 directory beside the private meta copy; and `score` parses ONLY
#     that file for such a task. No untrusted uid can open, reach or signal
#     anything on that path.
# --------------------------------------------------------------------------- #

_PROC_ACCMODE_WRITE = (os.O_WRONLY, os.O_RDWR)


def _pipe_writer_pids(ino: int) -> list[int]:
    """PIDs other than ours holding a WRITE end of pipe inode ``ino``."""
    me = os.getpid()
    target = f"pipe:[{ino}]"
    found: list[int] = []
    try:
        pids = [int(x) for x in os.listdir("/proc") if x.isdigit()]
    except OSError:
        return found
    for pid in pids:
        if pid == me:
            continue
        fd_dir = f"/proc/{pid}/fd"
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            continue
        for fd in fds:
            try:
                if os.readlink(f"{fd_dir}/{fd}") != target:
                    continue
                flags = 0
                with open(f"/proc/{pid}/fdinfo/{fd}") as fh:
                    for line in fh:
                        if line.startswith("flags:"):
                            flags = int(line.split()[1], 8)
                            break
            except (OSError, ValueError, IndexError):
                continue
            if (flags & os.O_ACCMODE) in _PROC_ACCMODE_WRITE:
                found.append(pid)
                break
    return found


def _has_exited(proc: subprocess.Popen) -> bool:
    """True once ``proc`` has exited, WITHOUT reaping it (waitid WNOWAIT)."""
    if proc.returncode is not None:
        return True
    try:
        info = os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    except (ChildProcessError, AttributeError, OSError):
        return proc.poll() is not None
    return info is not None


def _kill_capture_survivors(state: dict) -> int:
    """After the eval command's exit, SIGKILL its process group and every process
    still holding the write end of its stdout pipe (an escaped survivor).
    Returns how many pipe holders were killed."""
    proc = state.get("proc")
    if proc is not None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
    ino = state.get("pipe_ino")
    if not ino:
        return 0
    killed = 0
    for pid in _pipe_writer_pids(ino):
        try:
            os.kill(pid, signal.SIGKILL)
            killed += 1
        except OSError:
            pass
    return killed


def _uid_pids(uid: int) -> list[int]:
    """Live (non-zombie) processes whose real, effective or saved uid is ``uid``."""
    out: list[int] = []
    try:
        pids = [x for x in os.listdir("/proc") if x.isdigit()]
    except OSError:
        return out
    for pid in pids:
        try:
            with open(f"/proc/{pid}/status") as fh:
                text = fh.read()
        except OSError:
            continue
        m = re.search(r"^Uid:\s+(\d+)\s+(\d+)\s+(\d+)", text, re.M)
        z = re.search(r"^State:\s+([A-Z])", text, re.M)
        if m and uid in (int(m.group(1)), int(m.group(2)), int(m.group(3))) \
                and not (z and z.group(1) in ("Z", "X")):
            out.append(int(pid))
    return out


_KILL_ALL_SNIPPET = (
    "import os, signal\n"
    "try:\n    os.kill(-1, signal.SIGKILL)\n"
    "except OSError:\n    pass\n"
)


def _kill_uid_everywhere(uid: int, gid: int, timeout: float = 5.0) -> int:
    """SIGKILL every process of ``uid`` -- kill(-1) from a process running AS
    that uid, which the kernel applies to the whole task list atomically with
    respect to fork -- until none is left. Returns the number still alive
    (0 = clean; -1 = not attempted: not root, or uid 0)."""
    if uid == 0 or os.geteuid() != 0:
        return -1
    deadline = time.time() + timeout
    while True:
        try:
            subprocess.run([sys.executable, "-I", "-S", "-c", _KILL_ALL_SNIPPET],
                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=10,
                           **_drop_privileges_kwargs(uid, gid))
        except Exception:
            pass
        left = _uid_pids(uid)
        if not left or time.time() > deadline:
            return len(left)
        time.sleep(0.02)


class _ChildWakeup:
    """SIGCHLD -> a byte on a pipe the select loop watches, so an eval's exit is
    handled the instant it happens rather than at the next poll interval.
    Main thread only; a no-op elsewhere (the poll interval still applies)."""

    def __init__(self) -> None:
        self.fd = None
        self._w = None
        self._old_handler = None
        self._old_wakeup = None
        if threading.current_thread() is not threading.main_thread():
            return
        try:
            r, w = os.pipe()
            os.set_blocking(r, False)
            os.set_blocking(w, False)
            self._old_handler = signal.signal(signal.SIGCHLD, lambda *_a: None)
            self._old_wakeup = signal.set_wakeup_fd(w)
            self.fd, self._w = r, w
        except (ValueError, OSError):
            self.close()

    def drain(self) -> None:
        if self.fd is None:
            return
        try:
            while os.read(self.fd, 512):
                pass
        except OSError:
            pass

    def close(self) -> None:
        if self._w is not None:
            try:
                signal.set_wakeup_fd(self._old_wakeup if self._old_wakeup is not None else -1)
            except (ValueError, OSError):
                pass
        if self._old_handler is not None:
            try:
                signal.signal(signal.SIGCHLD, self._old_handler)
            except (ValueError, OSError):
                pass
        for fd in (self.fd, self._w):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        self.fd = self._w = self._old_handler = self._old_wakeup = None


def _trusted_result_dir(task_meta: Path) -> Path:
    """Root-only home of the per-(label, seed) result files: beside the private
    meta copy (test.sh makes its parent 0700 root), shared by run-evals, which
    creates it, and score, which reads it."""
    return Path(task_meta).parent / "results"


def _trusted_result_path(result_dir: Path, label: str, seed: int) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(label)) or "cmd"
    return result_dir / f"{safe}__seed{int(seed)}.result"


def _untrusted_sidecar(result_path: Path) -> Path:
    """Where run-untrusted records each invocation for this (label, seed)."""
    return Path(str(result_path) + ".untrusted")


def _prepare_trusted_result_dir(task_meta: Path) -> tuple[Path | None, str]:
    """Create the result dir (0700, owned by the verifier) or say why not."""
    if os.geteuid() != 0:
        return None, ("harbor_trusted_result REFUSED: the verifier is not root, so no "
                      "uid split can protect the channel; every record will be refused")
    parent = Path(task_meta).parent
    try:
        pst = os.lstat(parent)
    except OSError as exc:
        return None, f"harbor_trusted_result REFUSED: {parent}: {exc}"
    if not stat.S_ISDIR(pst.st_mode) or pst.st_uid != 0 or pst.st_mode & 0o022:
        return None, (f"harbor_trusted_result REFUSED: {parent} is not a root-owned "
                      f"directory closed to other writers (mode {oct(pst.st_mode & 0o7777)})")
    rdir = _trusted_result_dir(task_meta)
    shutil.rmtree(rdir, ignore_errors=True)
    os.mkdir(rdir, 0o700)
    os.chmod(rdir, 0o700)
    return rdir, (f"trusted result channel: {rdir} (0700 root); eval commands get "
                  "MLSBENCH_RESULT_FILE and `score` parses only that file")


_TRUSTED_RESULT_MAX_BYTES = 1 << 20


def _read_trusted_result(task_meta: Path, label: str, seed: int) -> tuple[str | None, str | None]:
    """(text, None) from the root-only result file, or (None, why it was refused)."""
    rdir = _trusted_result_dir(task_meta)
    path = _trusted_result_path(rdir, label, seed)
    me = os.geteuid()
    try:
        dst = os.lstat(rdir)
        if not stat.S_ISDIR(dst.st_mode) or dst.st_uid != me or dst.st_mode & 0o077:
            return None, f"trusted result dir {rdir} is not private to the verifier"
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        return None, f"trusted result missing: {path.name} (the trusted phase never wrote MLSBENCH_RESULT_FILE)"
    except OSError as exc:
        return None, f"trusted result unreadable: {exc}"
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_uid != me or st.st_nlink != 1:
            return None, f"trusted result {path.name} is not a private regular file"
        data = os.read(fd, _TRUSTED_RESULT_MAX_BYTES + 1)
    finally:
        os.close(fd)
    if len(data) > _TRUSTED_RESULT_MAX_BYTES:
        return None, f"trusted result {path.name} exceeds {_TRUSTED_RESULT_MAX_BYTES} bytes"
    text = data.decode("utf-8", "replace")
    if not text.strip():
        return None, f"trusted result {path.name} is empty"
    return text, None


def _scored_text(task_meta: Path, config: dict, label: str, seed: int,
                 log_path: Path) -> tuple[str | None, str | None]:
    """The text the task's parser is given for one record, or (None, reason).

    `harbor_trusted_result`: only the root-only result file (stdout is ignored
    for metrics). Otherwise the captured stdout. Either way every
    `harbor_unique_markers` regex must match exactly one line."""
    if config.get("harbor_trusted_result"):
        text, err = _read_trusted_result(task_meta, label, seed)
        if text is None:
            return None, err
        # The editable code must have run through $MLSBENCH_RUN_UNTRUSTED: the
        # helper (root) appends one line per invocation to a sidecar only it
        # and the verifier can create (the result dir is 0700 root).
        side = _untrusted_sidecar(_trusted_result_path(_trusted_result_dir(task_meta), label, seed))
        try:
            sst = os.lstat(side)
            entries = side.read_text().splitlines() if stat.S_ISREG(sst.st_mode) \
                and sst.st_uid == os.geteuid() else []
        except OSError:
            entries = []
        if not entries:
            return None, ("harbor_trusted_result: no $MLSBENCH_RUN_UNTRUSTED invocation was "
                          "recorded for this run -- the editable code did not go through the "
                          "untrusted-phase helper")
        bad = [e for e in entries if "survivors=0" not in e.split()]
        if bad:
            return None, f"harbor_trusted_result: run-untrusted left survivors: {bad[0]}"
    else:
        text = log_path.read_text(errors="replace")
    for pat in config.get("harbor_unique_markers") or []:
        n = len(re.findall(str(pat), text, re.M))
        if n != 1:
            return None, (f"marker /{pat}/ appears {n} times; exactly one is required "
                          "(harbor_unique_markers)")
    return text, None


# --------------------------------------------------------------------------- #
# Privilege drop for the eval phase
#
# The eval commands run the SUBMISSION's own code — that is what they are for —
# so they must never run with the verifier's own privileges. A root eval shares
# the container the verifier mounts its scoring metadata into, and file modes
# alone do not contain it: neither test.sh's `chmod -R a-w` nor `chmod go-rwx`
# on the private copy stops root, which holds CAP_DAC_OVERRIDE.
#
# So before an eval wave starts we:
#   * seal the secret directories — /tests/meta, /solution and the private meta
#     copy — by chmod 0700 on the DIRECTORY ITSELF. One inode each, so it is
#     cheap and trivially restored afterwards, and a non-root process loses
#     traverse permission on the whole subtree regardless of what is inside;
#   * point MLSBENCH_TASK_DIR at the same throwaway copy /workspace/_task
#     already points at (_build_eval_task_dir), which carries the eval-time
#     resources and none of the scoring metadata;
#   * make everything the eval legitimately writes usable by an unprivileged
#     uid — the workspace package tree, SAVE_PATH/OUTPUT_DIR, HOME, the data
#     roots — and redirect the caches whose default location is not writable
#     (numba compiles next to site-packages, matplotlib into $HOME/.config);
#   * start the eval command as uid/gid 65534.
#
# The verifier itself keeps running as root, so `score` still reads the private
# meta copy and the sealed directories are restored when the evals are done.
#
# Opt-out: a task that genuinely needs a root eval sets
# ``"harbor_eval_unprivileged": false`` in its config.json. Absent that key the
# adapter looks at the eval scripts themselves: some tasks already run their
# untrusted part unprivileged with a FINER split than this one
# (`setpriv`, `chown`, `runuser`, `pkill -9 -u`, or an explicit
# `[ "$(id -u)" != 0 ]` abort), and taking root away from those breaks the
# mechanism that was already protecting them. A textual hit therefore keeps the
# status quo for that run, and says so in /logs/verifier/privdrop.txt.
#
# The SEAL is not part of that opt-out: a kept root eval still starts its
# untrusted part as some non-root uid, and that uid must not read /tests/meta
# either, so cmd_run_evals seals the secrets for every root verifier
# (_seal_secrets) whether or not it performs the drop itself.
# --------------------------------------------------------------------------- #

_EVAL_UID = 65534
_EVAL_GID = 65534

# Evidence in an eval script that the task manages privileges itself.
_ROOT_EVAL_MARKERS = re.compile(
    r"(?:^|[\s;(&|`\"'])setpriv(?:[\s;)&|\"']|$)"
    r"|(?:^|[\s;(&|`\"'])chown(?:[\s;)&|\"']|$)"
    r"|(?:^|[\s;(&|`\"'])runuser(?:[\s;)&|\"']|$)"
    r"|pkill\s+(?:-\w+\s+)*-u\b"
    r"|\bid\s+-u\b[^\n]*!=\s*\"?0"
    r"|needs a root container",
    re.M,
)

# An eval-side file that resolves INTO the verifier's meta directory. Such a
# task needs a privileged eval and the seal makes its lookup fail silently:
# os.path.exists("/tests/meta/dgp.py") returns False (not raises) for an
# unprivileged uid once /tests/meta is 0700, so a stager just reports "input
# not found". We do not flip the decision on this — reading the holdout from
# eval-time code is the very thing being closed — but the pattern is detected
# and written to the privdrop log, so the cause is visible rather than silent.
_META_REACH = re.compile(
    r"/tests/meta/|['\"]\.\.['\"]\s*,\s*['\"]meta['\"]|\.\./\.\./meta/"
)

# Env vars naming a directory the eval writes to or caches in. Whatever they
# point at is opened for the eval uid before the wave starts.
_RELAX_ENV_KEYS = (
    "HOME", "TMPDIR", "TORCH_HOME", "HF_HOME", "HUGGINGFACE_HUB_CACHE",
    "TRANSFORMERS_CACHE", "XDG_CACHE_HOME", "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR", "NUMBA_CACHE_DIR", "MPLCONFIGDIR",
    "DATASETS", "DATA_ROOT", "DATA_DIR", "TENSET_DATA_ROOT",
)
# Always opened when they exist: the container root's home (several tasks read
# a baked corpus from /root/data) and the three places a data dep can land.
#
# `/mlsb_data_root` is not an alias of `/data`, it is where `/data` POINTS. A
# self-contained bundle runs its dep's prepare script against
# `--data-root /mlsb_data_root` and then publishes the result as
# `ln -sfn /mlsb_data_root/<dep> /data/<name>`
# (scripts/harbor/selfcontained.py:296, scripts/harbor/bake_data.py:83). The
# walk below never follows a symlink -- deliberately -- so opening `/data`
# alone reaches the symlink and stops: neither the target directory nor
# anything under it changes mode, and `privdrop.txt` says so in a way nobody
# reads as a failure ("/data: 12/13 entries opened").
#
# Some upstream trainers put their run directory under the data root itself
# (`results_root: ${DATA_ROOT}/results` and friends), so the data roots have to
# be writable by the eval uid and not merely readable: otherwise such a run
# dies with `PermissionError: [Errno 13]` on a path under `/data` before it
# emits a single metric. The docstring above promises the eval can write
# `/data`, and that promise has to hold for every bundle carrying a data dep.
_RELAX_FIXED_ROOTS = ("/root", "/data", "/opt/mlsb-data", "/mlsb_data_root")
# Of those, the roots that hold baked INPUT data (and nothing a run must
# rewrite in place): their files are opened read-only, see _chmod_tree.
_DATA_FIXED_ROOTS = ("/data", "/opt/mlsb-data", "/mlsb_data_root")
_DATA_ENV_KEYS = ("DATASETS", "DATA_ROOT", "DATA_DIR", "TENSET_DATA_ROOT")


def _is_data_root(root: Path, data_roots: set) -> bool:
    return root in data_roots and not any(
        str(root) == h or str(root).startswith(h.rstrip("/") + "/")
        for h in ("/root", "/tmp", "/workspace", "/home"))


# A chmod of a file that lives in a lower image layer copies the WHOLE file up
# into the container's writable layer on overlayfs without `metacopy=on`, which
# is the common default. A baked training corpus can be tens of GB, so opening
# one for writing costs its whole size in disk and many minutes of I/O before
# the first eval command even starts — on a bundle whose declared storage
# barely covers the corpus once. Such a corpus is read-only input to the eval
# and is already world-readable, so the write bits buy nothing; above this size
# we grant read bits only.
_COPYUP_GUARD_BYTES = 64 * 1024 * 1024


def _chmod_tree(
    root: Path,
    *,
    dir_add: int,
    file_add: int,
    limit: int = 2_000_000,
    budget_sec: float = 120.0,
    max_depth: int | None = None,
    files_read_only: bool = False,
) -> str:
    """OR `dir_add` / `file_add` into the mode of every entry under *root*.

    This is `chmod -R a+rwX` (or `a+rX`) written out, with four differences
    that matter here: symlinks are never followed, the walk is bounded, a
    large already-readable regular file is not made writable (see
    `_COPYUP_GUARD_BYTES`), and `max_depth` can stop the descent — depth 1 is
    "the root and its direct entries, nothing below". `files_read_only`
    grants a FILE only missing read (and x) bits, never write: on overlayfs
    every chmod of a lower-layer file copies the whole file up, so opening a
    baked dataset (MVTec, COCO) for writing file by file costs its full size in
    I/O and blew the time budget on local docker, leaving the tree half-opened.
    Directories still get `dir_add`, so the eval can create new files anywhere.
    A baked ImageNet-scale tree is ~1.3M inodes; the
    stat walk itself is a few seconds, but the caps mean a pathological tree
    degrades to "partially opened, and it says so in the log" rather than to a
    verifier that never starts.
    """
    t0 = time.time()
    seen = 0
    changed = 0
    truncated = False
    guarded = 0

    def _apply(path: str, is_dir: bool) -> None:
        nonlocal changed, guarded
        try:
            st = os.lstat(path)
        except OSError:
            return
        if stat.S_ISLNK(st.st_mode):
            return
        mode = stat.S_IMODE(st.st_mode)
        add = dir_add if is_dir else (file_add & 0o555 if files_read_only else file_add)
        want = mode | add
        if want == mode:
            return
        if (not is_dir and stat.S_ISREG(st.st_mode)
                and st.st_size > _COPYUP_GUARD_BYTES
                and (mode & 0o004)
                and (want & ~mode) & 0o222 == (want & ~mode)):
            # Only write bits are missing, the file is already world-readable,
            # and it is big enough that the copy-up would dominate the run.
            guarded += 1
            return
        try:
            os.chmod(path, want)
            changed += 1
        except OSError:
            pass

    if not root.exists():
        return f"{root}: absent"
    seen = 1
    _apply(str(root), root.is_dir())
    stack = [(str(root), 0)] if root.is_dir() else []
    while stack:
        cur, depth = stack.pop()
        try:
            entries = list(os.scandir(cur))
        except OSError:
            continue
        for entry in entries:
            seen += 1
            if seen > limit or (time.time() - t0) > budget_sec:
                truncated = True
                stack = []
                break
            try:
                is_dir = entry.is_dir(follow_symlinks=False)
            except OSError:
                is_dir = False
            _apply(entry.path, is_dir)
            if is_dir and (max_depth is None or depth + 1 < max_depth):
                stack.append((entry.path, depth + 1))
    note = f"{root}: {changed}/{seen} entries opened in {time.time() - t0:.1f}s"
    if guarded:
        note += (
            f" [{guarded} large read-only file(s) left unwritable — overlayfs "
            f"copy-up guard, > {_COPYUP_GUARD_BYTES // (1024 * 1024)} MiB]"
        )
    return note + " [TRUNCATED — deeper entries keep their modes]" if truncated else note


def _seal_dir(path: Path) -> tuple[str, int] | None:
    """chmod 0700 the directory itself; return (path, previous mode) to restore.

    Sealing the directory rather than its contents is deliberate: it is one
    chmod instead of a recursive walk over a tree that can hold a whole
    holdout corpus, it cannot be defeated by a file we forgot, and undoing it
    is a single syscall — which matters because /tests and /solution are bind
    mounts of the bundle on the host, and a bundle left unreadable to the
    invoking user would break the next run.
    """
    try:
        prev = stat.S_IMODE(os.stat(str(path)).st_mode)
    except OSError:
        return None
    try:
        os.chmod(str(path), 0o700)
    except OSError:
        return None
    return str(path), prev


def _restore_modes(saved: list) -> None:
    for path, mode in saved:
        try:
            os.chmod(path, mode)
        except OSError:
            pass


def _eval_side_files(eval_root: Path):
    """Every file the eval side can execute — not just the entry scripts.

    A family that factors its settings into a thin per-setting wrapper plus one
    shared harness keeps its privilege management in the harness, so reading
    only the `cmd` targets would miss it."""
    if not eval_root.exists():
        return []
    return [p for p in sorted(eval_root.rglob("*"))
            if p.is_file() and p.suffix in (".sh", ".py")]


def _unprivileged_eval_plan(config: dict, scripts) -> tuple[tuple[int, int] | None, str]:
    """(uid, gid) to run the eval commands as, plus the reason, or (None, why)."""
    declared = config.get("harbor_eval_unprivileged")
    if config.get("harbor_trusted_result"):
        return None, ("config.json sets harbor_trusted_result=true: the eval keeps root, "
                      "runs its untrusted phase through $MLSBENCH_RUN_UNTRUSTED and reports "
                      "through the root-only MLSBENCH_RESULT_FILE")
    if declared is not None and not bool(declared):
        return None, "config.json sets harbor_eval_unprivileged=false"
    if os.geteuid() != 0:
        return None, f"verifier already unprivileged (euid={os.geteuid()})"
    if declared is None:
        for script in scripts:
            try:
                text = Path(script).read_text(errors="ignore")
            except OSError:
                continue
            hit = _ROOT_EVAL_MARKERS.search(text)
            if hit:
                return None, (
                    f"{Path(script).name} manages privileges itself "
                    f"({hit.group(0).strip()!r}); leaving the eval as root — set "
                    "harbor_eval_unprivileged=true in config.json to override"
                )
    uid = int(config.get("harbor_eval_uid", _EVAL_UID) or _EVAL_UID)
    gid = int(config.get("harbor_eval_gid", uid) or uid)
    if uid == 0:
        return None, "config.json pins harbor_eval_uid=0"
    why = "config.json sets harbor_eval_unprivileged=true" if declared else \
        "default: no eval script manages privileges itself"
    return (uid, gid), why


def _drop_privileges_kwargs(uid: int, gid: int) -> dict:
    """subprocess kwargs that start the child unprivileged.

    No external binary: `setpriv` is not in every base image and uid 65534
    needs no passwd entry either way. 3.9+ has user=/group=/extra_groups; the
    3.8 images (Ubuntu 20.04, NGC pytorch:22.04-py3) take preexec_fn.
    """
    if sys.version_info >= (3, 9):
        return {"user": uid, "group": gid, "extra_groups": []}

    def _demote() -> None:  # pragma: no cover - runs in the forked child
        os.setgroups([])
        os.setgid(gid)
        os.setuid(uid)

    return {"preexec_fn": _demote}


def _open_scratch_for_reading(scratch_dir: Path, uid: int, gid: int) -> Path:
    """Make the budget-check scratch readable, and hand the child a writable tmp.

    budget_check.py legitimately reads its siblings here — config.json for the
    active test_cmd, edits/, scripts/ — and mkdtemp created the directory 0700,
    so without this the unprivileged child cannot open its own entry point
    ([Errno 13], rc=2, every submission rejected). Writes stay blocked: the
    tree gets a+rX with w cleared, and a single child-owned _tmp/ is handed
    over as TMPDIR.
    """
    for path in [scratch_dir, *scratch_dir.rglob("*")]:
        try:
            mode = stat.S_IMODE(path.stat().st_mode)
            os.chmod(path, (mode | 0o555) & ~0o222 if path.is_dir()
                     else (mode | 0o444) & ~0o222)
        except OSError:
            pass
    tmp = scratch_dir / "_tmp"
    try:
        tmp.mkdir(exist_ok=True)
        os.chown(tmp, uid, gid)
        os.chmod(tmp, 0o700)
    except OSError:
        return scratch_dir
    return tmp


def _secret_dirs(task_meta: Path) -> tuple[Path, ...]:
    """Every verifier-side directory eval-time code must not read.

    /tests/meta carries the staged holdout/ (dgp.py, gold labels, reference
    answers, netlists, ...), parser.py, score_spec.py and the leaderboard;
    /solution carries the oracle; the scoring assets carry frozen target
    metadata; task_meta is the private copy test.sh made of /tests/meta."""
    return (Path("/tests/meta"), Path("/solution"),
            Path("/tests/mlsbench_src/mlsbench/scoring/assets"),
            task_meta, task_meta.parent)


def _seal_secrets(task_meta: Path) -> tuple[list, list[str]]:
    """chmod 0700 every secret directory; return (restore list, log lines).

    Called for EVERY eval run whose verifier is root, not only when this
    process performs the privilege drop itself. A task kept on a root eval
    (``harbor_eval_unprivileged: false``, or a script matching
    _ROOT_EVAL_MARKERS) still runs its untrusted part as some non-root uid --
    it drops privileges itself, with setpriv/runuser/preexec -- and before
    that uid could read /tests/meta, because nothing sealed it: the
    seal lived only in the adapter-drop branch. Measured on
    chip-macro-placement-search: the setpriv'd search policy could open
    /tests/meta/netlists/*.pkl and decode layouts offline, outside the metered
    budget. Root keeps CAP_DAC_OVERRIDE, so the trusted, root half of such a
    task (stagers, oracles, graders reading $MLSBENCH_TASK_DIR or /tests/meta)
    reads exactly as before; only a non-root process loses traverse.
    """
    notes: list[str] = []
    sealed: list = []
    for secret in _secret_dirs(task_meta):
        entry = _seal_dir(secret)
        if entry is not None:
            sealed.append(entry)
            notes.append(f"sealed {entry[0]} (was {entry[1]:04o})")
        elif secret.exists():
            # The seal is the half of this that hides the gold labels; the uid
            # alone only stops writes. A venue that mounts /tests read-only
            # would land here, so say it in a word that greps.
            notes.append(
                f"SEAL FAILED {secret} — could not chmod 0700; eval-time code "
                "can still READ it"
            )
    return sealed, notes


def _prepare_unprivileged_eval(
    *,
    task_meta: Path,
    workspace_root: Path,
    out_dir: Path,
    test_cmds: list,
    uid: int,
    gid: int,
) -> tuple[list, list[str]]:
    """Seal the secrets, open what the eval writes. Returns (restore, log lines)."""
    sealed, notes = _seal_secrets(task_meta)

    # The eval scripts themselves must stay readable — bash re-reads the script
    # as it executes it — and so must the frozen harness beside them.
    notes.append(_chmod_tree(Path("/tests/eval"), dir_add=0o555, file_add=0o444))

    # Everything the eval legitimately writes.
    notes.append(_chmod_tree(workspace_root, dir_add=0o777, file_add=0o666))
    save_path = out_dir / "save"
    save_path.mkdir(parents=True, exist_ok=True)
    notes.append(_chmod_tree(save_path, dir_add=0o777, file_add=0o666))
    try:
        os.chmod("/tmp", 0o1777)
    except OSError:
        pass
    # /tmp being 1777 is not enough: a file a ROOT build step left there is
    # still root-owned and mode 644, and a library that opens a fixed path
    # under /tmp at import time then dies for the eval uid before the task's
    # own code runs. Simulator and logging libraries do exactly this: a
    # module-scope `logging.FileHandler("/tmp/<lib>.log")` opened during an
    # image build check leaves a root-owned file behind, and every later eval
    # then hits `PermissionError: [Errno 13]` on it as uid 65534 and extracts
    # no metrics at all. Like the /data case above, that is the drop's own
    # stated guarantee failing rather than a task quirk, so the top-level
    # entries of /tmp are opened as well.
    #
    # Depth 1 on purpose. Build steps also leave pip target trees under /tmp
    # (a `--target` dir for a data-prep script's deps is tens of thousands of
    # small files); nothing at eval time writes into those, and chmodding them
    # would copy every one up into the container layer for nothing. What
    # breaks things is a top-level file, and that is what this opens.
    notes.append(_chmod_tree(Path("/tmp"), dir_add=0o777, file_add=0o666,
                             max_depth=1))

    roots = {Path(p) for p in _RELAX_FIXED_ROOTS}
    data_roots = {Path(p) for p in _DATA_FIXED_ROOTS}
    for tc in test_cmds:
        for key, value in (tc.get("env") or {}).items():
            if str(key) in _RELAX_ENV_KEYS and str(value).startswith("/"):
                roots.add(Path(str(value)))
                if str(key) in _DATA_ENV_KEYS:
                    data_roots.add(Path(str(value)))
    for key in _RELAX_ENV_KEYS:
        value = os.environ.get(key, "")
        if value.startswith("/"):
            roots.add(Path(value))
            if key in _DATA_ENV_KEYS:
                data_roots.add(Path(value))
    for root in sorted(roots):
        if root.exists() and str(root) not in ("/", "/tests", "/solution", "/workspace"):
            # Data roots are read-only INPUT: their files only need to be
            # readable (baked files already are, so nothing is copied up);
            # their directories stay writable for runs that put outputs there.
            notes.append(_chmod_tree(root, dir_add=0o777, file_add=0o666,
                                     files_read_only=_is_data_root(root, data_roots)))

    for path in _eval_side_files(Path("/tests/eval")):
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        # Comment lines are skipped: every A-pattern eval script in this repo
        # *describes* /tests/meta in its header, and a note that fires on all
        # of them is a note nobody reads.
        hit = None
        for line in text.splitlines():
            if line.lstrip().startswith("#"):
                continue
            hit = _META_REACH.search(line)
            if hit:
                break
        if hit:
            notes.append(
                f"NOTE {path} resolves into the verifier's meta dir "
                f"({hit.group(0)!r}); that lookup now FAILS SILENTLY under the "
                "seal. A task whose eval legitimately needs a verifier-side "
                "resource must set harbor_eval_unprivileged=false and meter the "
                "access itself (see compile-schedule-search-policy)."
            )
            break

    notes.append(f"eval commands will run as uid={uid} gid={gid}")
    return sealed, notes



# --------------------------------------------------------------------------- #
# Trusted code in the writable workspace
#
# Under the privilege drop the workspace tree is opened a+rwX for uid 65534,
# because evals legitimately write into it. That also made every FIXED code
# file in it -- evaluators, scorers, harness modules a task's mid_edit staged
# next to the editable file -- writable by the untrusted phase of an eval, and
# every eval command runs as that same uid, so an earlier phase could overwrite
# a trusted evaluator that a later phase then runs. The
# diff guard runs once, before the evals, so it never saw it.
#
# So after the workspace is opened, every existing NON-EDITABLE code file
# loses its write bits (it stays root-owned, so uid 65534 cannot chmod it back
# or rewrite it in place), and its (sha256, owner, inode) is recorded. Before
# each eval wave and once after the last one the record is re-checked: a file
# that changed, was deleted, or was replaced (unlink + recreate in the still
# writable directory gives a new inode and a 65534 owner) invalidates the run
# -- every (setting, seed) becomes an error record, reward 0. Data files keep
# their write bits: many evals rewrite them.
# --------------------------------------------------------------------------- #

_LOCK_CODE_SUFFIXES = (".py", ".sh", ".bash", ".so", ".pyx", ".pxd", ".js", ".lua", ".jl", ".R", ".r")


def _editable_workspace_files(config: dict, workspace_root: Path) -> set:
    out = set()
    for f in config.get("files", []) or []:
        if f.get("edit"):
            out.add(str((workspace_root / str(f.get("filename", ""))).resolve()))
    return out


def _lock_workspace_code(workspace_root: Path, config: dict) -> dict:
    editable = _editable_workspace_files(config, workspace_root)
    locked: dict = {}
    for dirpath, dirnames, filenames in os.walk(str(workspace_root)):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"
                       and not os.path.islink(os.path.join(dirpath, d))]
        for name in filenames:
            path = os.path.join(dirpath, name)
            try:
                st = os.lstat(path)
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode) or path in editable:
                continue
            if not (name.endswith(_LOCK_CODE_SUFFIXES) or st.st_mode & 0o111):
                continue
            try:
                os.chmod(path, stat.S_IMODE(st.st_mode) & ~0o222)
                locked[path] = (_hash_file(path), st.st_uid, st.st_ino)
            except OSError:
                continue
    return locked


def _verify_workspace_code(locked: dict) -> list:
    problems = []
    for path, (sha, uid, ino) in locked.items():
        try:
            st = os.lstat(path)
        except OSError:
            problems.append(f"deleted: {path}")
            continue
        if st.st_ino != ino or st.st_uid != uid:
            problems.append(f"replaced (inode/owner changed): {path}")
        elif stat.S_IMODE(st.st_mode) & 0o222:
            problems.append(f"made writable: {path}")
        elif _hash_file(path) != sha:
            problems.append(f"modified: {path}")
    return problems


def _apply_unprivileged_env(env: dict, uid: int, gid: int) -> None:
    """Point the caches whose default target is not writable by *uid* somewhere
    that is, without moving the ones that already resolve into a writable HOME
    (a baked HF / torch cache under /root is exactly what we just opened up)."""
    scratch = "/tmp/mlsbench-eval-tmp"
    try:
        os.makedirs(scratch, exist_ok=True)
        os.chmod(scratch, 0o1777)
    except OSError:
        scratch = "/tmp"
    home = env.get("HOME") or "/root"
    if not os.path.isdir(home) or not os.access(home, os.W_OK):
        home = scratch
    env["HOME"] = home
    env.setdefault("TMPDIR", scratch)
    # numba compiles its cache next to the source file — under site-packages,
    # which an unprivileged uid cannot write — and matplotlib writes a font
    # cache. Both are silent-failure-then-crash for a task that uses them.
    env.setdefault("NUMBA_CACHE_DIR", os.path.join(scratch, "numba"))
    env.setdefault("MPLCONFIGDIR", os.path.join(scratch, "mpl"))
    for key in ("NUMBA_CACHE_DIR", "MPLCONFIGDIR"):
        try:
            os.makedirs(env[key], exist_ok=True)
            os.chmod(env[key], 0o1777)
        except OSError:
            pass
    env.setdefault("XDG_CACHE_HOME", os.path.join(home, ".cache"))
    # tiktoken resolves its BPE cache to $TIKTOKEN_CACHE_DIR, else
    # $DATA_GYM_CACHE_DIR, else `tempfile.gettempdir()/data-gym-cache` — and
    # `tempfile.gettempdir()` reads TMPDIR, which we just moved. So an image
    # that warmed the cache at its DEFAULT location (commonly
    # /tmp/data-gym-cache) loses it here, and tiktoken then reaches for the
    # network on an offline node. An image that names a directory the eval uid
    # cannot create fails harder: `tiktoken.get_encoding(...)` raises
    # `PermissionError: [Errno 13]` and takes the command down with it. Point
    # the variable at a warm cache when one exists, and at the scratch dir
    # otherwise so the makedirs at least succeeds.
    _tiktoken = env.get("TIKTOKEN_CACHE_DIR") or env.get("DATA_GYM_CACHE_DIR")
    if not (_tiktoken and os.path.isdir(_tiktoken) and os.access(_tiktoken, os.R_OK)):
        warm = next(
            (c for c in (os.path.join(d, "data-gym-cache")
                         for d in ("/tmp", "/root", "/opt"))
             if os.path.isdir(c) and os.listdir(c)),
            None,
        )
        env["TIKTOKEN_CACHE_DIR"] = warm or os.path.join(scratch, "data-gym-cache")
        env["DATA_GYM_CACHE_DIR"] = env["TIKTOKEN_CACHE_DIR"]
        if warm is None:
            try:
                os.makedirs(env["TIKTOKEN_CACHE_DIR"], exist_ok=True)
                os.chmod(env["TIKTOKEN_CACHE_DIR"], 0o1777)
            except OSError:
                pass


# Baseline reference implementations never reach a task-dir copy that
# submission-authored code can read.
#
# `tasks/<t>/edits/` is two things at once. It is the task SCAFFOLD — mid_edit.py,
# custom_template.py, and the harness_*/ *_src.py modules an eval script copies
# into place — and it is the BASELINE LIBRARY: every `<name>.edit.py` is the OPS
# payload that materialises one published method into the editable region, and a
# handful of tasks add helper modules those payloads import. Both copies built
# below are exposed to code the SUBMISSION wrote: the eval wave runs the agent's
# package code with MLSBENCH_TASK_DIR and /workspace/_task pointing at
# _build_eval_task_dir's copy, and budget_check.py imports the agent's editable
# file, so whatever that file runs at module scope runs with
# _copy_task_meta_for_budget's copy exposed the same way, immediately before the
# eval wave. A submission that can read the strongest baseline's OPS can exec it
# and score as that baseline without doing any science, so it does not get to.
#
# There IS one legitimate reader, and a grep for `.edit.py` does not find it:
# budget_check.py sizes the parameter cap by applying every baseline's ops to
# the template, and 115 of them resolve those paths out of config.json's
# baselines[].edit_ops, so the literal string never appears in the file. Both
# predicates below are therefore content-derived rather than a task allowlist —
# a new task in that shape keeps working — and both are computed from files WE
# stage, never from anything the submission can write.
_BASELINE_IMPL_NAME = re.compile(
    r".+\.edit\.py$"                    # every baseline's OPS payload
    r"|.+\.reference\.py$"              # theory-flip-flop: measured reference variant
    r"|.+\.skyline\.py$"                # theory-flip-flop: reference skyline
    r"|^reference_arms\.py$"            # mlsys-mla-decode-absorption, mlsys-moe-dropless-dispatch
    r"|^baseline_reference_ops\.py$"    # cv-flowmaps-training
    r"|^paper_losses\.py$"              # cv-flowmaps-training (imported by the above)
)


def _ignore_baseline_impls(_dirpath, names):
    return [n for n in names if _BASELINE_IMPL_NAME.match(n)]


def _budget_check_loads_baseline_ops(task_meta: Path) -> bool:
    """True when the task's own budget_check.py needs the baselines' OPS
    payloads: it applies each baseline's ops to the template, counts parameters
    and caps the submission at 1.05x the largest. Two tasks name
    `<name>.edit.py` outright (dl-shift-invariance-penalty,
    spec-drafter-block-head); the other 113 resolve config.json's
    baselines[].edit_ops, so the literal filename never appears."""
    try:
        text = (task_meta / "budget_check.py").read_text(errors="replace")
    except OSError:
        return False
    return ".edit.py" in text or "edit_ops" in text


def _eval_reruns_budget_check(task_meta: Path) -> bool:
    """True when an eval script re-runs budget_check.py against the task dir
    THIS copy provides (`python "${TASK_META}/budget_check.py" --checkpoint …`,
    where TASK_META resolves to MLSBENCH_TASK_DIR — the llm-linattn /
    llm-looped / llm-segmem / moe / vla-smolvla families). Combined with the
    predicate above that is the only route from the eval copy to a payload: 26
    tasks. Everything else that reads one under tasks/<t>/scripts/ is an
    offline audit_*/mem_check/cpu_check diagnostic that no test_cmd invokes
    (and that the renderer usually strips before the bundle is written)."""
    scripts = task_meta / "scripts"
    if not scripts.is_dir():
        return False
    for p in scripts.rglob("*"):
        if not p.is_file() or "__pycache__" in p.parts:
            continue
        try:
            if "budget_check" in p.read_text(errors="replace"):
                return True
        except OSError:
            continue
    return False


def _copy_task_meta_for_budget(
    task_meta: Path, scratch_dir: Path,
    effective_test_cmds: list[dict] | None = None,
) -> None:
    scratch_dir.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "budget_check.py"):
        src = task_meta / name
        if src.exists():
            shutil.copy2(src, scratch_dir / name)
    # The budget check is the one thing that legitimately needs the payloads.
    ignore = None if _budget_check_loads_baseline_ops(task_meta) else _ignore_baseline_impls
    # Same holdout filter as the eval-time copy: budget_check.py runs the
    # agent's editable module, unprivileged, with this copy as its task dir.
    ignore = _ignore_hidden(ignore, task_meta, _holdout_eval_exposure(task_meta)[0])
    for name in ("edits", "scripts"):
        src = task_meta / name
        if src.exists():
            shutil.copytree(src, scratch_dir / name, dirs_exist_ok=True, ignore=ignore)
    # budget_check.py derives the agent model's hyperparameters from this
    # config.json's test_cmds (active_test_cmd -> cmd -> expand_script_argv). For
    # an oracle run the eval cmd is replaced by the strongest baseline's cmd, and
    # that substitution MUST be reflected here too — otherwise the budget check
    # counts the agent model under the ORIGINAL (large) eval-script
    # hyperparameters and wrongly rejects the oracle baseline (all-zero TS
    # oracle). Native MLSBench runs the budget check against the
    # baseline-substituted task config; mirror that.
    if effective_test_cmds is not None:
        cfg_path = scratch_dir / "config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            cfg["test_cmds"] = effective_test_cmds
            cfg_path.write_text(json.dumps(cfg, indent=2))


def _install_budget_legacy_links(scratch_dir: Path, workspace_root: Path) -> list[Path]:
    links: list[Path] = []
    for dst in {workspace_root / "_task", Path("/workspace/_task")}:
        try:
            if dst.exists() or dst.is_symlink():
                if dst.is_dir() and not dst.is_symlink():
                    shutil.rmtree(dst)
                else:
                    dst.unlink()
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(scratch_dir, dst, target_is_directory=True)
            links.append(dst)
        except OSError:
            continue
    return links


def _remove_budget_legacy_links(links: list[Path]) -> None:
    for link in links:
        try:
            if link.is_symlink():
                link.unlink()
        except OSError:
            pass


# What the eval-time /workspace/_task copy carries. This is the same set the
# budget-check scratch copy gets (_copy_task_meta_for_budget) — modulo the
# baseline-implementation filter above, which the two copies apply under
# different conditions — for the same reason: run scripts that re-run the
# budget check on the trained checkpoint
# (`python /workspace/_task/budget_check.py --checkpoint ...`, the
# llm-linattn / llm-looped / llm-segmem families) read budget_check.py, the
# config.json it derives hyperparameters from, and the edits/ scaffold. If any
# of those is absent from the copy, such a re-check reports INVALID after a
# clean training run, so this set has to stay complete.
_EVAL_TASK_FILES = ("config.json", "budget_check.py", "budget_block.md")
_EVAL_TASK_DIRS = ("scripts", "data", "third_party", "edits", "configs")
# "configs" is not cosmetic: the mlsys-serving-* tasks resolve their
# per-setting cluster configs as "$MLSBENCH_TASK_DIR/configs/<name>.json",
# and under the privilege drop MLSBENCH_TASK_DIR IS this copy (the real
# task_meta is sealed), so an unstaged configs/ is a missing file in every
# setting.


# Files the adapter staged from host-only holdout/<task>/ (A-pattern generator,
# references, verifier-only data) are listed in tests/meta/holdout_paths.json.
# They land under the same meta subdirs as the task's own eval resources
# (holdout/<t>/data -> meta/data, holdout/<t>/scripts -> meta/scripts), so a
# plain copy of _EVAL_TASK_DIRS would re-export them, world-readable under the
# privilege drop, to the uid that runs editable code. Only a holdout file that
# the task's OWN eval-time files actually reference is exported; the rest
# stay in the sealed meta, where only the root score step reads them.
_HOLDOUT_MANIFEST = "holdout_paths.json"
_TASKDIR_DATA_REF = re.compile(r"(?:TASK_DIR\}?\"?|_task)/data\b")


def _holdout_eval_exposure(task_meta: Path) -> tuple[set[str], set[str], bool]:
    """(hidden, exposed, have_manifest): meta-relative holdout files under
    _EVAL_TASK_DIRS that the eval-time copy leaves out / still carries."""
    try:
        listed = json.loads((task_meta / _HOLDOUT_MANIFEST).read_text())
    except (OSError, ValueError):
        return set(), set(), False
    holdout = {str(p) for p in listed
               if isinstance(p, str) and p.split("/", 1)[0] in _EVAL_TASK_DIRS}
    texts: list[str] = []
    for sub in _EVAL_TASK_DIRS:
        root = task_meta / sub
        if not root.is_dir():
            continue
        for f in root.rglob("*"):
            rel = f.relative_to(task_meta).as_posix()
            if (rel in holdout or not f.is_file() or f.is_symlink()
                    or "__pycache__" in f.parts or f.stat().st_size > 4_000_000):
                continue
            try:
                texts.append(f.read_text(errors="ignore"))
            except OSError:
                continue
    for name in _EVAL_TASK_FILES:
        try:
            texts.append((task_meta / name).read_text(errors="ignore"))
        except OSError:
            pass
    blob = "\n".join(texts)
    data_dir_ref = bool(_TASKDIR_DATA_REF.search(blob))
    exposed = {h for h in holdout
               if Path(h).name in blob or (h.startswith("data/") and data_dir_ref)}
    return holdout - exposed, exposed, True


def _ignore_hidden(base, task_meta: Path, hidden):
    """copytree ignore = *base* plus the meta-relative paths in *hidden*."""
    if not hidden:
        return base

    def ignore(dirpath, names):
        out = set(base(dirpath, names)) if base else set()
        rel_dir = Path(dirpath).relative_to(task_meta).as_posix()
        out.update(n for n in names if f"{rel_dir}/{n}" in hidden)
        return out
    return ignore


def _build_eval_task_dir(task_meta: Path, hidden: set[str] | frozenset = frozenset()) -> Path:
    """Throwaway dir exposing ONLY the eval-time resources (_EVAL_TASK_DIRS +
    _EVAL_TASK_FILES) that eval wrappers reach via /workspace/_task or
    $MLSBENCH_TASK_DIR. It deliberately EXCLUDES the scoring metadata
    (parser.py / score_spec.py / leaderboard.csv / pristine_manifest.json) that
    cmd_score imports: eval runs agent-authored package code as root, so any
    path it can reach is writable (chmod a-w doesn't stop root), and exposing
    the real task_meta would let a submission overwrite the parser/spec before
    the score phase. The real task_meta stays at its unexposed random /tmp path.
    cmd_score never reads this copy, so nothing a submission does to it can
    change its score.

    `edits/` is staged for its SCAFFOLD only — the baseline reference
    implementations in it are filtered out (_BASELINE_IMPL_NAME), because the
    same agent-authored code that reaches this copy could otherwise exec the
    strongest baseline's OPS and score as that baseline. The exception is the
    26 tasks whose eval script re-runs budget_check.py against this copy: that
    check derives the parameter cap FROM the baselines, and without them it
    reports INVALID after a clean training run."""
    d = Path(tempfile.mkdtemp(prefix="mlsbench-evaltask-"))
    ignore = (
        None
        if (_budget_check_loads_baseline_ops(task_meta)
            and _eval_reruns_budget_check(task_meta))
        else _ignore_baseline_impls
    )
    for name in _EVAL_TASK_FILES:
        src = task_meta / name
        if src.exists():
            shutil.copy2(src, d / name)
    ignore = _ignore_hidden(ignore, task_meta, hidden)
    for sub in _EVAL_TASK_DIRS:
        src = task_meta / sub
        if src.exists():
            shutil.copytree(src, d / sub, dirs_exist_ok=True, ignore=ignore)
    return d


def _run_budget_check(
    *,
    task_meta: Path,
    workspace_root: Path,
    pkg_dir: Path,
    out_dir: Path,
    label: str,
    seed: int,
    env: dict[str, str],
    effective_test_cmds: list[dict] | None = None,
    drop: tuple[int, int] | None = None,
) -> dict | None:
    if not (task_meta / "budget_check.py").exists():
        return None
    log_path = out_dir / f"{label}__seed{seed}__budget_check.log"
    # Use the same hardened interpreter as test.sh — MLSBENCH_VERIFIER_PYTHON
    # is exported by test.sh after PATH reset; falls back to sys.executable
    # (which is itself a hardened interpreter since we run under test.sh).
    python_bin = os.environ.get("MLSBENCH_VERIFIER_PYTHON") or sys.executable
    # Note: NOT using -I here because budget_check.py may legitimately need
    # PYTHONPATH from the package env (e.g. to import the model defined under
    # /workspace/<pkg>/). We rely on the env dict being verifier-controlled
    # — test.sh stripped agent-planted PYTHONPATH before this script runs.
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "-", label)[:64] or "test"
    scratch_dir = Path(tempfile.mkdtemp(prefix=f"mlsbench-budget-{safe_label}-{seed}-"))
    legacy_links: list[Path] = []
    with log_path.open("w") as fh:
        try:
            _copy_task_meta_for_budget(task_meta, scratch_dir, effective_test_cmds)
            legacy_links = _install_budget_legacy_links(scratch_dir, workspace_root)
            budget_env = env.copy()
            budget_env["TMPDIR"] = str(scratch_dir)
            budget_env["MLSBENCH_TASK_DIR"] = str(scratch_dir)
            # budget_check.py is the sharpest instance of "the verifier executes
            # agent code": 93 of the 140 checks in this repo load the editable
            # file with spec_from_file_location to count parameters, so whatever
            # the agent wrote at module scope runs here — after the guard has
            # passed, in the verifier's own process tree. Same drop as the eval.
            # See docs/security/budget_check_privilege.md.
            #
            # This holds for keep-root tasks too: their eval stays
            # root because the TASK splits privileges inside its own scripts,
            # but budget_check.py is run by THIS process, not by those scripts,
            # so nothing else would ever drop it -- and 12 keep-root checks
            # import the submission (cv-pytorch-*, cv-ssl-*, dl-distill-*,
            # dl-dynamics-grokking-transition). Module scope would then run as
            # root, after the seal, where root still reads /tests/meta.
            drop_kwargs: dict = {}
            bdrop = drop
            if bdrop is None and os.geteuid() == 0:
                bdrop = (_EVAL_UID, _EVAL_GID)
            if bdrop is not None:
                budget_env["TMPDIR"] = str(
                    _open_scratch_for_reading(scratch_dir, bdrop[0], bdrop[1])
                )
                if drop is None:
                    # The keep-root eval never opened HOME/caches for an
                    # unprivileged uid; give the check its own writable ones.
                    _apply_unprivileged_env(budget_env, *bdrop)
                    budget_env["TMPDIR"] = str(scratch_dir / "_tmp")
                    budget_env["HOME"] = str(scratch_dir / "_tmp")
                    budget_env["XDG_CACHE_HOME"] = str(scratch_dir / "_tmp" / ".cache")
                drop_kwargs = _drop_privileges_kwargs(*bdrop)
                fh.write(f"[budget] running budget_check.py as uid={bdrop[0]} gid={bdrop[1]}"
                         f"{' (root eval kept; the check is dropped regardless)' if drop is None else ''}\n")
                fh.flush()
            proc = subprocess.run(
                [python_bin, str(scratch_dir / "budget_check.py")],
                cwd=str(pkg_dir),
                env=budget_env,
                stdout=fh,
                stderr=subprocess.STDOUT,
                timeout=120,
                check=False,
                **drop_kwargs,
            )
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            fh.write("\n[BUDGET CHECK TIMEOUT] budget_check.py took >120s\n")
            rc = 124
        except Exception as exc:
            fh.write(f"\n[BUDGET CHECK ERROR] {exc}\n")
            rc = 125
        finally:
            _remove_budget_legacy_links(legacy_links)
            shutil.rmtree(scratch_dir, ignore_errors=True)
    return {"rc": rc, "log": str(log_path)}


def _eval_log_path(out_dir: Path, label: str, seed: int) -> Path:
    return out_dir / f"{label}__seed{seed}.log"


def _write_error_record(
    out_dir: Path,
    entry: dict,
    seed: int,
    message: str,
    rc: int,
) -> dict:
    log_path = _eval_log_path(out_dir, entry["label"], seed)
    log_path.write_text(message.rstrip() + "\n")
    return {
        "seed": seed,
        "rc": rc,
        "log": str(log_path),
        "elapsed": 0.0,
    }


def _finish_process_record(state: dict, seed: int, rc: int | None = None) -> dict:
    if rc is None:
        rc = state["proc"].returncode
    if rc is None:
        rc = 124
    elapsed = time.time() - state["start"]
    # _stop_capture normally closed the pipe + capture at exit; this is the
    # belt-and-braces for the paths that build a record without it.
    _stop_capture(state)
    return {
        "seed": seed,
        "rc": rc,
        "log": str(state["log_path"]),
        "elapsed": elapsed,
    }


# ---------------------------------------------------------------------------
# Withheld inputs (config.json ``"ephemeral_inputs": true``)
# ---------------------------------------------------------------------------
# Some tasks withhold data from the agent's editable code -- hidden training
# labels, held-out targets, oracle tables.  Their ``edits/mid_edit.py``
# materializes that data per (ENV, SEED) by importing the task's holdout
# ``dgp.py``, and the task's FIXED wrapper (``scripts/fixed_entry.py``) loads
# the blobs into memory and unlinks them BEFORE the editable module is
# imported.  Natively the harness re-stages the blobs host-side before every
# test command (``mlsbench.agent.input_stager``).  Under Harbor nothing did,
# in an earlier version the bundle's eval found no blobs and the opted-in
# tasks could not be scored.
#
# The verifier regenerates the run's blobs itself -- it is the only actor that
# can read the sealed meta copy, where the adapter stages ``dgp.py`` beside
# ``edits/mid_edit.py`` -- and hands them to the (unprivileged) eval by one of
# two transports, chosen by config.json ``"ephemeral_inputs_transport"``:
#
#   "disk"  (default) the blobs are written into the workspace exactly as the
#           native stager does and the wrapper reads + unlinks them
#           (``--inputs-glob``).  Every existing ephemeral task's scripts work
#           unchanged.  The blobs exist on the shared filesystem between the
#           write and the wrapper's unlink, so a sibling eval of the same
#           concurrent wave could in principle read them in that window.
#   "stdin" the blobs are piped to the eval's stdin as ONE JSON object
#           ``{op-file: content}`` and the eval env carries
#           ``MLSBENCH_WITHHELD_STDIN=1``; the wrapper consumes them with
#           ``--inputs-json-stdin``.  No withheld byte touches the shared
#           filesystem, so a sibling eval cannot read them.  This is the
#           earlier public-side fix (``apply.py --emit-json``) moved to
#           the verifier, where it is privdrop-safe: the generator stays in
#           the sealed meta, only the payload crosses the pipe.
#
# ``mid_edit.py`` is run by a child interpreter under ``-S -B`` from the sealed
# meta with the frozen harness on its path, so its ``load_holdout_module``
# resolves the sealed ``dgp.py`` (never a ``__file__``-derived guess) and no
# bytecode is written next to the sealed sources.
_WITHHELD_TRANSPORTS = ("disk", "stdin")

_WITHHELD_DRIVER = r"""
import json, runpy, sys
mid, src = sys.argv[1], sys.argv[2]
sys.dont_write_bytecode = True
if src and src not in sys.path:
    sys.path.insert(0, src)
ns = runpy.run_path(mid, run_name="__mlsbench_withheld_inputs__")
out = {}
for op in list(ns.get("OPS") or []):
    if not isinstance(op, dict) or op.get("op") != "create":
        continue
    rel = str(op.get("file", ""))
    if not rel or rel.endswith(".py"):
        continue  # the scaffold itself, never a withheld blob
    out[rel] = str(op.get("content", ""))
sys.stdout.write(json.dumps(out))
"""


def _withheld_transport(config: dict) -> str | None:
    """Return the withheld-input transport for this task, or None when the
    task does not opt in (``ephemeral_inputs`` absent or false)."""
    if not config.get("ephemeral_inputs"):
        return None
    transport = str(config.get("ephemeral_inputs_transport") or "disk").lower()
    if transport not in _WITHHELD_TRANSPORTS:
        raise ValueError(
            f"config.json ephemeral_inputs_transport={transport!r}; "
            f"expected one of {_WITHHELD_TRANSPORTS}"
        )
    return transport


def _mlsbench_src_for_driver() -> str:
    """Where the child that runs mid_edit finds ``mlsbench``: the bundle's own
    frozen harness under Harbor (cmd_score's rule), else wherever this process
    imports it from (unit tests, native experiments)."""
    frozen = Path("/tests/mlsbench_src")
    if (frozen / "mlsbench" / "__init__.py").is_file():
        return str(frozen)
    try:
        import mlsbench  # noqa: F401
    except Exception:
        return ""
    return str(Path(mlsbench.__file__).resolve().parents[1])


def _regenerate_withheld_inputs(
    task_meta: Path,
    label: str,
    seed: int,
    *,
    timeout: int = 900,
) -> dict[str, str]:
    """Run the task's mid_edit from the sealed meta with ENV/SEED set and return
    its non-``.py`` create ops as ``{workspace-relative file: content}``."""
    mid = task_meta / "edits" / "mid_edit.py"
    if not mid.is_file():
        raise FileNotFoundError(f"{mid} is missing; the task declares ephemeral_inputs")
    env = os.environ.copy()
    env["ENV"] = str(label)
    env["SEED"] = str(seed)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-S", "-B", "-c", _WITHHELD_DRIVER, str(mid), _mlsbench_src_for_driver()],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(task_meta),
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"mid_edit exited {proc.returncode} while regenerating withheld inputs "
            f"for {label!r} seed {seed}:\n{proc.stderr[-4000:]}"
        )
    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"mid_edit emitted malformed withheld-input JSON: {exc}") from exc
    if not isinstance(data, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in data.items()
    ):
        raise RuntimeError("withheld inputs must be a JSON object mapping file -> text")
    return data


def _withheld_dst(workspace_root: Path, rel: str) -> Path:
    """Resolve an op file to a workspace path the way the native stager does:
    a first component naming a workspace package dir (normalized match), else
    a literal join.  Rejects traversal."""
    rel = _safe_rel_path(rel)
    parts = rel.split("/")
    if len(parts) > 1 and workspace_root.is_dir():
        norm = _normalize_pkg_name(parts[0])
        for child in workspace_root.iterdir():
            if child.is_dir() and _normalize_pkg_name(child.name) == norm:
                return child.joinpath(*parts[1:])
    return workspace_root / rel


def _stage_withheld_on_disk(
    payload: dict[str, str],
    workspace_root: Path,
    drop: tuple[int, int] | None,
) -> list[Path]:
    """Disk transport: write the blobs into the workspace (atomic rename) so the
    task's wrapper can read and unlink them as the eval uid.  Returns the paths
    installed, for the post-wave backstop scrub."""
    staged: list[Path] = []
    for rel, content in payload.items():
        dst = _withheld_dst(workspace_root, rel)
        parent = dst.parent
        created_parent = not parent.exists()
        parent.mkdir(parents=True, exist_ok=True)
        if created_parent or drop is not None:
            try:
                os.chmod(parent, 0o777)
            except OSError:
                pass
        tmp = parent / f".{dst.name}.{os.getpid()}.stage.tmp"
        with open(tmp, "w") as fh:
            fh.write(content)
        os.chmod(tmp, 0o666 if drop is not None else 0o644)
        os.replace(tmp, dst)
        staged.append(dst)
    return staged


def _scrub_withheld(paths: list[Path]) -> int:
    """Backstop: unlink whatever the wrapper did not (a crashed eval).  Returns
    the number of files that were still present."""
    left = 0
    for path in paths:
        try:
            os.unlink(path)
            left += 1
        except FileNotFoundError:
            pass
        except OSError:
            left += 1
    return left


def _feed_stdin(write_fd: int, payload: bytes, proc: subprocess.Popen, deadline: float) -> None:
    """Write ``payload`` to the eval's stdin pipe and close it.  Non-blocking
    with a deadline so an eval that never reads its stdin cannot park this
    thread (and its fd) forever."""
    view = memoryview(payload)
    off = 0
    try:
        os.set_blocking(write_fd, False)
        while off < len(view):
            if proc.poll() is not None or time.time() > deadline:
                break
            try:
                _, ready, _ = select.select([], [write_fd], [], 1.0)
            except (OSError, ValueError):
                break
            if not ready:
                continue
            try:
                off += os.write(write_fd, view[off:off + 65536])
            except BlockingIOError:
                continue
            except OSError:
                break
    finally:
        try:
            os.close(write_fd)
        except OSError:
            pass


def _run_eval_wave(
    *,
    tasks: list[dict],
    assignments: list[str | None],
    task_meta: Path,
    workspace_root: Path,
    default_pkg: str,
    out_dir: Path,
    drop: tuple[int, int] | None = None,
    eval_task_dir: Path | None = None,
    withheld_stdin: dict[tuple[int, int], bytes] | None = None,
    result_dir: Path | None = None,
) -> dict[tuple[int, int], dict]:
    wakeup = _ChildWakeup()
    try:
        results = _run_eval_wave_inner(
            tasks=tasks, assignments=assignments, task_meta=task_meta,
            workspace_root=workspace_root, default_pkg=default_pkg, out_dir=out_dir,
            drop=drop, eval_task_dir=eval_task_dir, withheld_stdin=withheld_stdin,
            result_dir=result_dir, wakeup=wakeup,
        )
    finally:
        wakeup.close()
    if drop is not None:
        # Nothing of a dropped wave may outlive it: a survivor could otherwise
        # tamper with the next wave's outputs or the files `score` reads.
        left = _kill_uid_everywhere(*drop)
        for rec in results.values():
            rec["wave_end_uid_survivors"] = left
    return results


def _run_eval_wave_inner(
    *,
    tasks: list[dict],
    assignments: list[str | None],
    task_meta: Path,
    workspace_root: Path,
    default_pkg: str,
    out_dir: Path,
    drop: tuple[int, int] | None,
    eval_task_dir: Path | None,
    withheld_stdin: dict[tuple[int, int], bytes] | None,
    result_dir: Path | None,
    wakeup: "_ChildWakeup",
) -> dict[tuple[int, int], dict]:
    timeout_secs = int(
        EVAL_TIME_FACTOR * max(
            _parse_time_to_seconds(task["entry"]["tc"].get("time", "1:00:00"))
            for task in tasks
        )
    ) + WAVE_GRACE_SEC
    deadline = time.time() + timeout_secs
    running: list[dict] = []
    results: dict[tuple[int, int], dict] = {}

    for task, gpu_devices in zip(tasks, assignments):
        entry = task["entry"]
        seed = int(task["seed"])
        log_path = _eval_log_path(out_dir, entry["label"], seed)
        pkg_dir = _package_dir(workspace_root, default_pkg, entry["tc"])
        env = _eval_env(
            task_meta=task_meta,
            out_dir=out_dir,
            workspace_root=workspace_root,
            pkg_dir=pkg_dir,
            tc=entry["tc"],
            seed=seed,
            unprivileged=drop,
            task_dir=eval_task_dir if drop is not None else None,
        )
        if gpu_devices:
            env["CUDA_VISIBLE_DEVICES"] = gpu_devices
            env["NVIDIA_VISIBLE_DEVICES"] = gpu_devices
        if result_dir is not None:
            rpath = _trusted_result_path(result_dir, entry["label"], seed)
            try:
                os.unlink(rpath)
            except FileNotFoundError:
                pass
            try:
                os.unlink(_untrusted_sidecar(rpath))
            except FileNotFoundError:
                pass
            os.close(os.open(rpath, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                             | getattr(os, "O_NOFOLLOW", 0), 0o600))
            env["MLSBENCH_RESULT_FILE"] = str(rpath)
        else:
            env.pop("MLSBENCH_RESULT_FILE", None)
        # The eval's stdout is a PIPE the verifier drains, not an open file the
        # eval (and any process it leaks) would inherit a writable handle to —
        # see the "Trusted capture" section. The capture file is root-owned.
        cap_fh = open(log_path, "wb")
        read_fd, write_fd = os.pipe()
        os.set_blocking(read_fd, False)
        # Withheld inputs over stdin (see the block above _run_eval_wave): the
        # payload crosses a pipe the parent writes from a helper thread, and
        # the wrapper's --inputs-json-stdin drains it before any editable code
        # runs.  Nothing else in this verifier gives the eval a stdin.
        stdin_payload = (withheld_stdin or {}).get((entry["idx"], seed))
        stdin_r = stdin_w = None
        popen_stdin: dict = {}
        if stdin_payload is not None:
            env["MLSBENCH_WITHHELD_STDIN"] = "1"
            stdin_r, stdin_w = os.pipe()
            popen_stdin = {"stdin": stdin_r}
        t_start = time.time()
        try:
            proc = subprocess.Popen(
                ["bash", str(entry["script"])],
                cwd=str(pkg_dir),
                env=env,
                stdout=write_fd,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                **popen_stdin,
                **(_drop_privileges_kwargs(*drop) if drop is not None else {}),
            )
        except Exception as exc:
            os.close(read_fd)
            os.close(write_fd)
            for fd in (stdin_r, stdin_w):
                if fd is not None:
                    os.close(fd)
            cap_fh.write(f"[ERROR] failed to start eval command: {exc}\n".encode())
            cap_fh.close()
            results[(entry["idx"], seed)] = {
                "seed": seed,
                "rc": 127,
                "log": str(log_path),
                "elapsed": time.time() - t_start,
            }
            continue
        # Only the child (and anything it spawns) keeps the write end; the parent
        # keeps only the read end, so the pipe reaches EOF as soon as the last
        # writer — honest or escaped — closes it.
        os.close(write_fd)
        if stdin_payload is not None:
            os.close(stdin_r)  # only the child keeps the read end
            threading.Thread(
                target=_feed_stdin,
                args=(stdin_w, stdin_payload, proc, deadline),
                daemon=True,
            ).start()
        running.append({
            "entry": entry,
            "seed": seed,
            "proc": proc,
            "cap_fh": cap_fh,
            "read_fd": read_fd,
            "pipe_ino": os.fstat(read_fd).st_ino,
            "start": t_start,
            "log_path": log_path,
        })

    while running and time.time() < deadline:
        # Poll for exits FIRST, and freeze an exited command's capture before
        # doing any more reading. A survivor that waits for the eval to vanish
        # and then writes cannot have its bytes captured, because the pipe read
        # end is closed the moment we observe the exit — as long as we observe it
        # promptly, hence the short interval below rather than a 0.5 s sleep.
        still_running: list[dict] = []
        for state in running:
            if not _has_exited(state["proc"]):
                still_running.append(state)
            else:
                # Freeze the capture FIRST and reap only afterwards: the reap
                # makes /proc/<shell> vanish, which is itself a trigger a
                # survivor can key on, and scanning /proc for pipe holders
                # takes milliseconds. Survivors are killed once the read end
                # is closed (their later writes get EPIPE). What a survivor
                # wrote between the shell's exit and this drain is still
                # captured -- the race this narrows but cannot close (see
                # "Survivor-process forgery" above).
                _stop_capture(state)
                rc = state["proc"].wait()
                killed = _kill_capture_survivors(state)
                rec = _finish_process_record(state, state["seed"], rc)
                if killed:
                    rec["stdout_survivors_killed"] = killed
                results[(state["entry"]["idx"], state["seed"])] = rec
        running = still_running
        if not running:
            break

        # Drain what the still-running commands have produced so the 64 KB pipe
        # buffer never fills and blocks them. select wakes us the instant any
        # pipe is readable OR the interval elapses, so an exit is noticed within
        # one short interval.
        live_fds = [s["read_fd"] for s in running if s.get("read_fd") is not None]
        watch = live_fds + ([wakeup.fd] if wakeup.fd is not None else [])
        if watch:
            try:
                ready, _, _ = select.select(watch, [], [], 0.1)
            except (OSError, ValueError):
                ready = live_fds
            ready_set = set(ready)
            if wakeup.fd is not None and wakeup.fd in ready_set:
                # A child exited: go straight back to the exit poll.
                wakeup.drain()
                continue
            for state in running:
                if state.get("read_fd") in ready_set:
                    _drain_captured(state)
        else:
            time.sleep(0.1)

    for state in running:
        _kill_capture_survivors(state)
        _stop_capture(state)
        try:
            with open(state["log_path"], "ab") as fh:
                fh.write(
                    f"\n[TIMEOUT] Command timed out after {timeout_secs} seconds.\n".encode()
                )
        except OSError:
            pass
        _kill_process_group(state["proc"].pid, timeout=30.0)
        try:
            state["proc"].wait(timeout=1)
        except Exception:
            pass
        results[(state["entry"]["idx"], state["seed"])] = _finish_process_record(
            state,
            state["seed"],
            124,
        )

    return results


def _parse_oracle_cmd_overrides(raw: str | None) -> list[dict]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid oracle cmd overrides JSON: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError("oracle cmd overrides must be a JSON list")

    out: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("oracle cmd override entries must be objects")
        cmd = str(item.get("cmd", "")).strip()
        env = item.get("env")
        if env is not None and not isinstance(env, dict):
            raise ValueError("oracle cmd override env must be an object")
        env = {str(k): str(v) for k, v in (env or {}).items()}
        if not cmd and not env:
            raise ValueError("oracle cmd override needs a non-empty cmd or env")
        override: dict = {}
        if cmd:
            override["cmd"] = cmd
        if env:
            override["env"] = env
        if "labels" in item:
            labels = item.get("labels") or [""]
            if not isinstance(labels, list):
                labels = [labels]
            out.extend({"label": str(label), **override} for label in labels)
        else:
            out.append({"label": str(item.get("label", "")), **override})
    return out


def _apply_oracle_cmd_overrides(
    test_cmds: list[dict],
    overrides: list[dict],
) -> list[dict]:
    result = [dict(tc) for tc in test_cmds]
    for override in overrides:
        label = override["label"]
        for entry in result:
            if not label or str(entry.get("label", "")) == label:
                if override.get("cmd"):
                    entry["cmd"] = override["cmd"]
                if override.get("env"):
                    # Baseline env (native cli exports baseline_config["env"])
                    # reaches the eval subprocess through tc["env"].
                    merged = dict(entry.get("env") or {})
                    merged.update(override["env"])
                    entry["env"] = merged
    return result


def cmd_run_evals(args: argparse.Namespace) -> int:
    task_meta = Path(args.task_meta)
    workspace_root = Path(args.workspace)
    default_pkg = _read_meta_text(task_meta, "package", "")
    eval_root = Path(args.eval_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = _load_task_config(task_meta)
    # Select an H200-specific command/env/compute override when the verifier
    # is actually running on H200 hardware.  This keeps Harbor's direct
    # sandbox behavior aligned with MLS-Bench's native ``compute_scale=0.5``
    # path and, importantly, applies the same materialized list to budget
    # checks, GPU packing, and the eval subprocesses.
    test_cmds = _effective_test_cmds(config)
    oracle_cmd_overrides = _parse_oracle_cmd_overrides(
        getattr(args, "oracle_cmd_overrides", None)
    )
    if oracle_cmd_overrides:
        test_cmds = _apply_oracle_cmd_overrides(test_cmds, oracle_cmd_overrides)
    seeds = _config_seeds(config)

    summary = [
        {"label": tc.get("label", tc.get("cmd", "test")), "hidden": bool(tc.get("hidden")), "logs": []}
        for tc in test_cmds
    ]
    records: dict[tuple[int, int], dict] = {}
    prepared: dict[int, dict] = {}
    for idx, tc in enumerate(test_cmds):
        cmd_rel = tc.get("cmd", "")
        label = tc.get("label", cmd_rel)
        # _safe_join rejects absolute paths, `..` traversal, and Windows
        # backslashes — without it, a hostile config could point at
        # /workspace/payload.sh and run agent-controlled code as verifier.
        try:
            script = Path(_safe_join(eval_root, cmd_rel))
        except ValueError as exc:
            entry = {"idx": idx, "tc": tc, "label": label}
            for seed in seeds:
                records[(idx, seed)] = _write_error_record(
                    out_dir,
                    entry,
                    seed,
                    f"[ERROR] unsafe_cmd_path: {exc}",
                    126,
                )
            continue
        if not script.exists():
            entry = {"idx": idx, "tc": tc, "label": label}
            for seed in seeds:
                records[(idx, seed)] = _write_error_record(
                    out_dir,
                    entry,
                    seed,
                    f"[ERROR] missing_script: {cmd_rel}",
                    127,
                )
            continue
        prepared[idx] = {"idx": idx, "tc": tc, "label": label, "script": script}

    # The eval commands run the submission's own code. Decide once, for the
    # whole run, whether they run as root (see the section above), then seal the
    # secrets and open what an unprivileged eval legitimately writes.
    drop, drop_reason = _unprivileged_eval_plan(
        config, _eval_side_files(eval_root) or [e["script"] for e in prepared.values()]
    )
    headline = f"unprivileged eval: {'yes' if drop else 'no'} — {drop_reason}"
    print(f"[privdrop] {headline}", flush=True)
    priv_notes = [headline]
    sealed: list = []
    locked_code: dict = {}
    integrity_notes: list = []
    if drop is not None:
        sealed, notes = _prepare_unprivileged_eval(
            task_meta=task_meta,
            workspace_root=workspace_root,
            out_dir=out_dir,
            test_cmds=test_cmds,
            uid=drop[0],
            gid=drop[1],
        )
        atexit.register(_restore_modes, sealed)
        priv_notes.extend(notes)
        locked_code = _lock_workspace_code(workspace_root, config)
        priv_notes.append(
            f"locked {len(locked_code)} non-editable code file(s) in the workspace "
            "(write bits removed; sha256/owner/inode re-checked before each wave "
            "and after the last)"
        )
    elif os.geteuid() == 0:
        # Root eval kept (opt-out or self-managed privileges). The task drops
        # privileges for its untrusted part itself, so the secrets still have
        # to be sealed before any of its commands start -- whoever drops.
        sealed, notes = _seal_secrets(task_meta)
        atexit.register(_restore_modes, sealed)
        priv_notes.append(
            "root eval kept: secrets sealed anyway (a task that drops "
            "privileges itself must not leave /tests/meta readable to its "
            "unprivileged children)"
        )
        priv_notes.extend(notes)
        if config.get("harbor_trusted_result"):
            # The untrusted phase runs under run-untrusted's uid, which cannot
            # write root-owned files anyway; the lock adds the (sha256, owner,
            # inode) re-check before each wave and after the last, so trusted
            # code in the workspace that changed between phases is an error.
            locked_code = _lock_workspace_code(workspace_root, config)
            priv_notes.append(
                f"locked {len(locked_code)} non-editable code file(s) in the workspace "
                "(trusted-result task: sha256/owner/inode re-checked before each wave "
                "and after the last)"
            )
    result_dir: Path | None = None
    if config.get("harbor_trusted_result"):
        result_dir, note = _prepare_trusted_result_dir(task_meta)
        priv_notes.append(note)
        print(f"[privdrop] {note}", flush=True)
    holdout_hidden, holdout_exposed, have_holdout_manifest = _holdout_eval_exposure(task_meta)
    if holdout_hidden:
        priv_notes.append(
            f"holdout: {len(holdout_hidden)} verifier-only file(s) kept out of the eval-time "
            "task copy (no eval script of the task reads them)")
    if holdout_exposed:
        priv_notes.append(
            f"holdout: {len(holdout_exposed)} file(s) the task's eval scripts read are in the "
            "eval-time task copy" + (
                " (root-only: this task keeps a root eval)" if drop is None else
                ", READABLE BY THE EVAL UID -- deliver them on stdin (ephemeral_inputs) to "
                "close this: " + ", ".join(sorted(holdout_exposed)[:8])))
    if not have_holdout_manifest and (task_meta / "dgp.py").exists():
        priv_notes.append(
            f"holdout: no tests/meta/{_HOLDOUT_MANIFEST}; holdout files under "
            f"{'/'.join(_EVAL_TASK_DIRS)} are exported unfiltered (re-stage the bundle)")
    (out_dir / "privdrop.txt").write_text("\n".join(priv_notes) + "\n")

    # Withheld inputs: regenerated per (label, seed) from the sealed meta and
    # delivered by the task's declared transport (block above _run_eval_wave).
    try:
        withheld_transport = _withheld_transport(config)
    except ValueError as exc:
        print(f"[withheld-inputs] {exc}", flush=True)
        (out_dir / "withheld_inputs.txt").write_text(f"ERROR {exc}\n")
        withheld_transport = "invalid"
    withheld_notes: list[str] = [
        f"transport: {withheld_transport or 'none (task does not withhold inputs)'}"
    ]

    grouped = _group_entries(test_cmds)
    devices = _visible_gpu_indices(task_meta, config)
    n_reserved = len(devices)

    for group_key in sorted(grouped.keys()):
        group_entries = [
            prepared[idx]
            for idx, _ in grouped[group_key]
            if idx in prepared
        ]
        if not group_entries:
            continue

        group_tasks = [
            {"entry": entry, "seed": seed}
            for entry in group_entries
            for seed in seeds
        ]

        schedulable: list[dict] = []
        for task in group_tasks:
            entry = task["entry"]
            seed = int(task["seed"])
            need = _task_gpu_need(task)
            if n_reserved > 0 and need > n_reserved:
                records[(entry["idx"], seed)] = _write_error_record(
                    out_dir,
                    entry,
                    seed,
                    (
                        "[ERROR] test_cmd compute requires "
                        f"{need} GPUs but only {n_reserved} reserved/visible"
                    ),
                    125,
                )
            else:
                schedulable.append(task)

        if not schedulable:
            continue

        batches = _partition_group_gpu_batches(schedulable, devices)
        if batches is None:
            for task in schedulable:
                entry = task["entry"]
                seed = int(task["seed"])
                records[(entry["idx"], seed)] = _write_error_record(
                    out_dir,
                    entry,
                    seed,
                    (
                        "[ERROR] unable to allocate GPUs for test_cmd "
                        f"with compute={_test_cmd_compute(entry['tc'])}"
                    ),
                    125,
                )
            continue

        for wave_tasks, assignments in batches:
            runnable_tasks: list[dict] = []
            runnable_assignments: list[str | None] = []
            wave_withheld_stdin: dict[tuple[int, int], bytes] = {}
            wave_staged: list[Path] = []
            for task, gpu_devices in zip(wave_tasks, assignments):
                entry = task["entry"]
                seed = int(task["seed"])
                pkg_dir = _package_dir(workspace_root, default_pkg, entry["tc"])
                env = _eval_env(
                    task_meta=task_meta,
                    out_dir=out_dir,
                    workspace_root=workspace_root,
                    pkg_dir=pkg_dir,
                    tc=entry["tc"],
                    seed=seed,
                    unprivileged=drop,
                )
                budget = _run_budget_check(
                    task_meta=task_meta,
                    workspace_root=workspace_root,
                    pkg_dir=pkg_dir,
                    out_dir=out_dir,
                    label=entry["label"],
                    seed=seed,
                    env=env,
                    effective_test_cmds=test_cmds,
                    drop=drop,
                )
                if budget and budget["rc"] != 0:
                    records[(entry["idx"], seed)] = _write_error_record(
                        out_dir,
                        entry,
                        seed,
                        f"[BUDGET CHECK FAILED]\nSee {budget['log']}",
                        int(budget["rc"]),
                    )
                    with (out_dir / "budget_violation.txt").open("a") as fh:
                        fh.write(
                            f"{entry['label']} seed {seed} failed budget_check.py; "
                            f"see {budget['log']}\n"
                        )
                    continue
                if withheld_transport is not None:
                    if withheld_transport == "invalid":
                        records[(entry["idx"], seed)] = _write_error_record(
                            out_dir, entry, seed,
                            "[ERROR] invalid ephemeral_inputs_transport in config.json", 125,
                        )
                        continue
                    try:
                        payload = _regenerate_withheld_inputs(task_meta, entry["label"], seed)
                    except Exception as exc:
                        records[(entry["idx"], seed)] = _write_error_record(
                            out_dir, entry, seed,
                            f"[ERROR] withheld inputs could not be regenerated: {exc}", 125,
                        )
                        withheld_notes.append(
                            f"{entry['label']} seed {seed}: REGENERATION FAILED: {exc}"
                        )
                        continue
                    if withheld_transport == "stdin":
                        wave_withheld_stdin[(entry["idx"], seed)] = json.dumps(payload).encode()
                        withheld_notes.append(
                            f"{entry['label']} seed {seed}: {len(payload)} blob(s) over stdin"
                        )
                    else:
                        try:
                            wave_staged.extend(
                                _stage_withheld_on_disk(payload, workspace_root, drop)
                            )
                        except Exception as exc:
                            records[(entry["idx"], seed)] = _write_error_record(
                                out_dir, entry, seed,
                                f"[ERROR] withheld inputs could not be staged: {exc}", 125,
                            )
                            withheld_notes.append(
                                f"{entry['label']} seed {seed}: STAGING FAILED: {exc}"
                            )
                            continue
                        withheld_notes.append(
                            f"{entry['label']} seed {seed}: {len(payload)} blob(s) on disk"
                        )
                runnable_tasks.append(task)
                runnable_assignments.append(gpu_devices)

            if not runnable_tasks:
                _scrub_withheld(wave_staged)
                continue

            # Expose the legacy /workspace/_task path that some eval wrappers
            # reference (humanoid: `python _task/scripts/...`; dllm:
            # `--data-path /workspace/_task/data/...`). Native MLSBench bind-mounts
            # the task dir there, but Harbor has no bind mounts. We point _task at
            # a MINIMAL throwaway copy (scripts/data/third_party only) rather than
            # task_meta, so eval-time agent code (running as root) cannot reach and
            # overwrite the scoring metadata (parser.py/score_spec.py/config.json)
            # before the score phase. The guard exempts the _task top-level dir and
            # runs as a separate pre-eval invocation, so this never affects the diff.
            eval_task_dir = _build_eval_task_dir(task_meta, holdout_hidden)
            if drop is not None:
                # mkdtemp made it 0700; the eval uid has to be able to read it,
                # and MLSBENCH_TASK_DIR points here for the same reason.
                _chmod_tree(eval_task_dir, dir_add=0o555, file_add=0o444)
            eval_task_links = _install_budget_legacy_links(eval_task_dir, workspace_root)
            tampered = _verify_workspace_code(locked_code) if locked_code else []
            if tampered:
                integrity_notes.extend(tampered)
                for task in runnable_tasks:
                    records[(task["entry"]["idx"], int(task["seed"]))] = _write_error_record(
                        out_dir, task["entry"], int(task["seed"]),
                        "[ERROR] trusted workspace code was modified by an earlier "
                        "eval phase; see workspace_integrity.txt", 125,
                    )
                _remove_budget_legacy_links(eval_task_links)
                shutil.rmtree(eval_task_dir, ignore_errors=True)
                _scrub_withheld(wave_staged)
                continue
            try:
                wave_results = _run_eval_wave(
                    tasks=runnable_tasks,
                    assignments=runnable_assignments,
                    task_meta=task_meta,
                    workspace_root=workspace_root,
                    default_pkg=default_pkg,
                    out_dir=out_dir,
                    drop=drop,
                    eval_task_dir=eval_task_dir,
                    withheld_stdin=wave_withheld_stdin or None,
                    result_dir=result_dir,
                )
            finally:
                _remove_budget_legacy_links(eval_task_links)
                shutil.rmtree(eval_task_dir, ignore_errors=True)
                # Backstop for the disk transport: the wrapper unlinks what it
                # loads; a crashed eval may not have, and the next wave must
                # not find a sibling's withheld blobs lying in the workspace.
                left = _scrub_withheld(wave_staged)
                if left:
                    withheld_notes.append(
                        f"backstop scrub removed {left} blob(s) an eval left on disk"
                    )
                if withheld_transport is not None:
                    (out_dir / "withheld_inputs.txt").write_text(
                        "\n".join(withheld_notes) + "\n"
                    )
            records.update(wave_results)
            for task in runnable_tasks:
                entry = task["entry"]
                seed = int(task["seed"])
                if (entry["idx"], seed) not in wave_results:
                    records[(entry["idx"], seed)] = _write_error_record(
                        out_dir,
                        entry,
                        seed,
                        "[ERROR] eval command produced no result",
                        125,
                    )

    if locked_code:
        late = _verify_workspace_code(locked_code)
        if late:
            # Tampering that persisted past the last wave: the run is voided,
            # whichever command did it.
            integrity_notes.extend(late)
            for (idx, seed) in list(records):
                entry = {"idx": idx, "tc": test_cmds[idx],
                         "label": test_cmds[idx].get("label", test_cmds[idx].get("cmd", ""))}
                records[(idx, seed)] = _write_error_record(
                    out_dir, entry, seed,
                    "[ERROR] trusted workspace code was modified during the eval; "
                    "see workspace_integrity.txt", 125,
                )
        (out_dir / "workspace_integrity.txt").write_text(
            f"{len(locked_code)} locked code file(s); "
            + (f"TAMPERED ({len(integrity_notes)}):\n" + "\n".join(integrity_notes[:100])
               if integrity_notes else "unchanged") + "\n"
        )

    for idx, _tc in enumerate(test_cmds):
        for seed in seeds:
            record = records.get((idx, seed))
            if record is not None:
                summary[idx]["logs"].append(record)

    _restore_modes(sealed)
    (out_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    return 0


# --------------------------------------------------------------------------- #
# Score: parse logs, aggregate, write reward
# --------------------------------------------------------------------------- #

def _aggregate_metrics(metrics_list: list[dict]) -> dict:
    try:
        from mlsbench.agent.tools import WorkspaceTools  # type: ignore[import-not-found]
        return WorkspaceTools._aggregate_metrics(metrics_list)
    except Exception:
        pass

    if not metrics_list:
        return {}
    if len(metrics_list) == 1:
        return metrics_list[0]

    collected: dict[str, list[float]] = {}
    for metrics in metrics_list:
        for key, value in metrics.items():
            try:
                collected.setdefault(key, []).append(float(value))
            except (TypeError, ValueError):
                pass

    aggregated: dict[str, float] = {}
    for key, values in collected.items():
        finite = [value for value in values if math.isfinite(value)]
        aggregated[key] = sum(finite) / len(finite) if finite else float("nan")
    return aggregated


def _has_real_metrics(record: dict) -> bool:
    for key, value in record.items():
        if (
            key in {"timestamp", "model", "is_final", "seed"}
            or str(key).startswith("elapsed_")
            or str(key).endswith("_std")
        ):
            continue
        if value in ("", None):
            continue
        return True
    return False


def _valid_seed_metric_records(per_seed_metrics: dict[int, dict]) -> list[dict]:
    return [metrics for _seed, metrics in sorted(per_seed_metrics.items()) if _has_real_metrics(metrics)]

def _completed_eval_record(log_info: dict) -> bool:
    """Never score a crashed/timed-out process, even if its log claims success."""
    # Do not accept False, the string "0", or a missing return code as evidence
    # of a completed run. This value comes from the trusted process supervisor.
    return type(log_info.get("rc")) is int and log_info["rc"] == 0


def cmd_score(args: argparse.Namespace) -> int:
    task_meta = Path(args.task_meta)
    out_dir = Path(args.out_dir)
    reward_out = Path(args.reward_out)
    reward_out.parent.mkdir(parents=True, exist_ok=True)

    # mlsbench src ships in the per-task tests/ dir (not in the base image —
    # the agent's shell would see it otherwise). Harbor mounts tests/ at
    # /tests/ only at verify time, so /tests/mlsbench_src is verifier-only.
    sys.path.insert(0, "/tests/mlsbench_src")
    sys.path.insert(0, str(task_meta))
    # Editable-install locations recovered by test.sh without running any .pth
    # code; APPENDED so nothing earlier on sys.path can be shadowed by them.
    for _extra in os.environ.get("MLSBENCH_VERIFIER_EXTRA_PATH", "").split(":"):
        if _extra and _extra not in sys.path:
            sys.path.append(_extra)
    try:
        # Pre-import & pin every mlsbench module we need INTO sys.modules
        # BEFORE we exec_module the task's parser.py. parser.py itself does
        # `sys.path.insert(0, PROJECT_ROOT / "src")` with PROJECT_ROOT
        # computed from its own __file__; for verifier mode that lands at
        # /tmp/<rand>/src which doesn't exist, but if a future change ever
        # makes it resolve to a real (and possibly different) mlsbench
        # package, the import cache here means parser still picks the
        # version we pinned. (Defense-in-depth against sys.path shadowing.)
        from mlsbench.scoring.evaluate import score_record, load_expanded_spec  # type: ignore[import-not-found]
        from mlsbench.scoring.anchors import BaselineAnchors  # type: ignore[import-not-found]
        from mlsbench.scoring.evaluation_settings import (
            evaluation_contract, check_evaluation_records, check_command_seed_coverage)
        import mlsbench.agent.parsers  # ensures the task parser inherits this version

        import __future__
        import types

        # Compile with PEP 563 (postponed annotation evaluation) so a task
        # parser written with PEP 585 generics (``-> tuple[str, dict]``,
        # ``list[float] = []``) also runs on the older container Pythons that
        # cannot subscript builtins at definition time (for example the
        # cleanrl image's Python 3.8).  Native scoring runs the parser in the
        # host's Python, so this difference only shows up inside Harbor.
        parser_path = task_meta / "parser.py"
        task_parser = types.ModuleType("task_parser")
        task_parser.__file__ = str(parser_path)
        code = compile(
            parser_path.read_text(),
            str(parser_path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        )
        exec(code, task_parser.__dict__)
    except Exception as exc:
        reward_out.write_text("0\n")
        (out_dir / "score_error.txt").write_text(f"import failed: {exc}\n")
        return 0

    config = json.loads((task_meta / "config.json").read_text())
    try:
        contract = evaluation_contract(task_meta, config)
    except (ValueError, KeyError, TypeError) as exc:
        reward_out.write_text("0\n")
        (out_dir / "score_error.txt").write_text(f"invalid evaluation settings: {exc}\n")
        return 0
    summary_path = out_dir / "eval_summary.json"
    if not summary_path.exists():
        reward_out.write_text("0\n")
        (out_dir / "score_error.txt").write_text("eval_summary.json missing\n")
        return 0
    summary = json.loads(summary_path.read_text())

    # Parse every log, aggregate per-seed metrics, then mean across seeds.
    test_cmd_by_label = {tc.get("label", tc["cmd"]): tc for tc in config.get("test_cmds", [])}
    per_seed_metrics: dict[int, dict] = {}
    evaluation_records = []
    for entry in summary:
        label = entry["label"]
        tc = test_cmd_by_label.get(label)
        if tc is None:
            continue
        # Every MLS-Bench parser.py defines `class Parser(OutputParser)` with
        # `parse(self, cmd_label, raw_output) -> ParseResult`.
        parser_inst = task_parser.Parser()
        # A dropped record is reported downstream as "expected one result, got
        # 0", which reads as a scheduling fault and sends the reader to the
        # wrong place -- the usual cause is the task's own parser raising on a
        # log whose command exited 0. So every (label, seed) the supervisor ran
        # produces a record; the ones with no usable metrics carry the reason.
        for log_info in entry.get("logs", []):
            seed = int(log_info.get("seed", -1))
            if "log" not in log_info:
                evaluation_records.append({"label": label, "seed": seed,
                                           "rc": log_info.get("rc"), "metrics": {},
                                           "error": "eval record carries no log path"})
                continue
            if not _completed_eval_record(log_info):
                evaluation_records.append({"label": label, "seed": seed,
                                           "rc": log_info.get("rc"), "metrics": {}})
                with (out_dir / "parse_errors.txt").open("a") as fh:
                    fh.write(f"{label} seed {seed}: refused metrics from incomplete eval "
                             f"(rc={log_info.get('rc')!r})\n")
                continue
            log_path = Path(log_info["log"])
            if not log_path.exists():
                evaluation_records.append({"label": label, "seed": seed,
                                           "rc": log_info.get("rc"), "metrics": {},
                                           "error": f"eval log is missing: {log_path}"})
                with (out_dir / "parse_errors.txt").open("a") as fh:
                    fh.write(f"{label} seed {seed}: eval log missing: {log_path}\n")
                continue
            log_text, refused = _scored_text(task_meta, config, label, seed, log_path)
            if log_text is None:
                evaluation_records.append({"label": label, "seed": seed,
                                           "rc": log_info.get("rc"), "metrics": {},
                                           "error": refused})
                with (out_dir / "parse_errors.txt").open("a") as fh:
                    fh.write(f"{label} seed {seed}: {refused}\n")
                continue
            try:
                parsed = parser_inst.parse(label, log_text)
            except Exception as exc:
                evaluation_records.append({"label": label, "seed": seed,
                                           "rc": log_info.get("rc"), "metrics": {},
                                           "error": f"parser raised {type(exc).__name__}: {exc}"})
                with (out_dir / "parse_errors.txt").open("a") as fh:
                    fh.write(f"{label} seed {seed}: {exc}\n")
                continue
            metrics = getattr(parsed, "metrics", None) or {}
            evaluation_records.append({"label": label, "seed": seed,
                                       "rc": log_info.get("rc"), "metrics": dict(metrics)})
            seed_metrics = per_seed_metrics.setdefault(seed, {})
            seed_metrics.update(metrics)
            if "elapsed" in log_info:
                try:
                    seed_metrics[f"elapsed_{label}"] = float(log_info["elapsed"])
                except (TypeError, ValueError):
                    pass

    if contract is not None:
        setting_summary = check_evaluation_records(contract, _config_seeds(config), evaluation_records)
        (out_dir / "setting_summary.json").write_text(json.dumps(setting_summary, indent=2))
        if not setting_summary["valid"]:
            reward_out.write_text("0\n")
            (out_dir / "score_error.txt").write_text(
                "incomplete evaluation settings:\n" + "\n".join(setting_summary["errors"]) + "\n")
            return 0
    else:
        # No explicit contract: still refuse to average an incomplete grid.
        # Dropping a failed run and averaging the survivors is not a neutral
        # default -- where the submission can make a run fail (a policy in a
        # child interpreter, a training script it controls), it turns the mean
        # over seeds into a best-of, and it costs the submission nothing.
        coverage = check_command_seed_coverage(
            list(test_cmd_by_label), _config_seeds(config), evaluation_records)
        (out_dir / "coverage_summary.json").write_text(json.dumps(coverage, indent=2))
        if not coverage["valid"]:
            reward_out.write_text("0\n")
            (out_dir / "score_error.txt").write_text(
                "incomplete evaluation runs:\n" + "\n".join(coverage["errors"]) + "\n")
            return 0

    valid_metrics = _valid_seed_metric_records(per_seed_metrics)
    if not valid_metrics:
        reward_out.write_text("0\n")
        (out_dir / "score_error.txt").write_text("no metrics extracted from logs\n")
        return 0

    mean_metrics = _aggregate_metrics(valid_metrics)

    # Score via mlsbench DSL against task's score_spec.py + anchors.
    anchors = BaselineAnchors(task_meta)
    spec = load_expanded_spec(task_meta, anchors)
    if spec is None:
        reward_out.write_text("0\n")
        (out_dir / "score_error.txt").write_text("score_spec missing or invalid\n")
        return 0

    combined = score_record(spec, mean_metrics, anchors)
    # combined_score is meant to be roughly in [0, 1]; clip defensively.
    if combined is None or combined != combined:  # NaN check
        combined = 0.0
    reward = max(0.0, min(1.0, float(combined)))

    reward_out.write_text(f"{reward}\n")
    (out_dir / "metrics.json").write_text(json.dumps({
        "combined_score": combined,
        "reward": reward,
        "mean_metrics": mean_metrics,
        "per_seed_metrics": per_seed_metrics,
        "evaluation_records": evaluation_records,
    }, indent=2))
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Site-tree manifest: the interpreters' import trees as they were at BUILD time
#
# The agent phase is root in this container for hours. Everything a trusted
# root process later imports -- the stdlib, site-packages (numpy, torch), and
# the startup hooks `site` runs (`sitecustomize.py`, `usercustomize.py`,
# `*.pth`) -- is therefore writable by the agent. `-S` on the verifier's own
# python closes the hooks for the verifier, but not for the eval scripts of
# the keep-root tasks, whose trusted root halves run a plain `python3`
# (measured: `python3 -P -B -E -s` still executes a planted sitecustomize.py
# and a planted .pth), and `-S` never closed a trojaned `numpy/__init__.py`.
#
# `site-manifest` (run INSIDE the freshly built image, post-build, by
# scripts/harbor/bake_interpreter_manifest.py) records every file under every
# directory any interpreter in the image puts on sys.path, plus the
# interpreter binaries and whether each root user-site dir exists.
# `site-check` (run by test.sh as root, before the guard and before any eval
# command) compares the live tree against it:
#   * a CHANGED or DELETED file, a changed interpreter, or a user-site dir
#     that did not exist at build  -> tampered: reward 0, reason logged;
#   * a NEW file                     -> removed (a pip-installed package, a
#     planted .pth or sitecustomize.py): the eval runs in the build-time
#     environment, as it does natively;
#   * new or changed bytecode under __pycache__ -> removed (it is regenerated
#     from the verified source; a planted .pyc with a matching header would
#     otherwise be loaded instead of the source).
# The manifest lives in tests/meta, which the agent never sees. A bundle
# without one gets the ELF interpreter check only.
# --------------------------------------------------------------------------- #

_SITE_INTERP_GLOBS = (
    "/usr/local/bin/python3*", "/usr/bin/python3*",
    "/opt/conda/bin/python3*", "/opt/conda/envs/*/bin/python3*",
    "/opt/miniconda3/bin/python3*", "/opt/miniconda3/envs/*/bin/python3*",
    "/opt/*/bin/python3*", "/root/.venv*/bin/python3*",
)
# sys.path entries that are NOT build-time environment: the agent's own
# workspace (editable installs point there; the diff guard owns it) and
# scratch space.
_SITE_EXCLUDE_PREFIXES = ("/workspace", "/tmp", "/logs", "/tests", "/solution")

_SITE_PROBE = (
    "import json,os,site,sys\n"
    "u=None\n"
    "try:\n    u=site.getusersitepackages()\nexcept Exception:\n    pass\n"
    "print('MLSB_SITE_PROBE '+json.dumps({'exe':os.path.realpath(sys.executable),"
    "'path':[p for p in sys.path if p],'user':u}))\n"
)


def _is_bytecode(path: str) -> bool:
    return "/__pycache__/" in path or path.endswith((".pyc", ".pyo"))


def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _site_tree_entries(dirs) -> dict:
    """{path: 'L:<target>' for a symlink | sha256 for a regular file}."""
    out: dict = {}
    for root in dirs:
        stack = [root]
        while stack:
            cur = stack.pop()
            try:
                it = list(os.scandir(cur))
            except OSError:
                continue
            for e in it:
                try:
                    if e.is_symlink():
                        out[e.path] = "L:" + os.readlink(e.path)
                    elif e.is_dir(follow_symlinks=False):
                        stack.append(e.path)
                    elif e.is_file(follow_symlinks=False):
                        out[e.path] = _hash_file(e.path)
                except OSError:
                    out[e.path] = "UNREADABLE"
    return out


def _collapse_dirs(dirs) -> list:
    keep: list = []
    for d in sorted(set(dirs), key=len):
        if not any(d == k or d.startswith(k.rstrip("/") + "/") for k in keep):
            keep.append(d)
    return sorted(keep)


def cmd_site_manifest(args: argparse.Namespace) -> int:
    import glob
    interps: dict = {}
    for pat in _SITE_INTERP_GLOBS:
        for cand in sorted(glob.glob(pat)):
            if cand.endswith("-config") or not os.access(cand, os.X_OK) or os.path.isdir(cand):
                continue
            real = os.path.realpath(cand)
            try:
                with open(real, "rb") as fh:
                    if fh.read(4) != b"\x7fELF":
                        continue
            except OSError:
                continue
            interps.setdefault(real, cand)
    dirs: list = []
    users: dict = {}
    probed: dict = {}
    for real, cand in sorted(interps.items()):
        try:
            out = subprocess.run([cand, "-E", "-c", _SITE_PROBE], capture_output=True,
                                 text=True, timeout=120, cwd="/").stdout
        except Exception:
            continue
        line = [l for l in out.splitlines() if l.startswith("MLSB_SITE_PROBE ")]
        if not line:
            continue
        info = json.loads(line[-1][len("MLSB_SITE_PROBE "):])
        probed[real] = cand
        for pth in info["path"]:
            rp = os.path.realpath(pth)
            if os.path.isdir(rp) and not rp.startswith(_SITE_EXCLUDE_PREFIXES) and rp != "/":
                dirs.append(rp)
        if info.get("user"):
            users[info["user"]] = os.path.isdir(info["user"])
    dirs = _collapse_dirs(dirs)
    # Compiled artifacts the BUILD left in the workspace (an in-tree extension
    # build), so the diff guard can tell them from planted ones.
    ws_compiled: dict = {}
    ws = Path(getattr(args, "workspace", "/workspace") or "/workspace")
    if ws.is_dir():
        for p in ws.rglob("*"):
            try:
                if (p.is_file() and not p.is_symlink() and "__pycache__" not in p.parts
                        and p.name.endswith((".so", ".pyd", ".pyc", ".pyo"))):
                    ws_compiled[p.relative_to(ws).as_posix()] = _hash_file(str(p))
            except OSError:
                continue
    manifest = {
        "version": 1,
        "workspace_compiled": ws_compiled,
        "interpreters": {real: _hash_file(real) for real in sorted(probed)},
        "dirs": dirs,
        "user_sites": users,
        "files": _site_tree_entries(dirs),
    }
    text = json.dumps(manifest, sort_keys=True)
    if args.out == "-":
        sys.stdout.write(text + "\n")
    else:
        Path(args.out).write_text(text + "\n")
    return 0


def cmd_site_check(args: argparse.Namespace) -> int:
    man = json.loads(Path(args.manifest).read_text())
    want = man.get("files", {})
    have = _site_tree_entries(man.get("dirs", []))
    tampered: list = []
    removed_new: list = []
    removed_pyc: list = []
    for real, sha in man.get("interpreters", {}).items():
        try:
            cur = _hash_file(real)
        except OSError:
            cur = "MISSING"
        if cur != sha:
            tampered.append(f"interpreter {real} changed")
    for path, sha in want.items():
        cur = have.get(path)
        if cur == sha:
            continue
        if _is_bytecode(path):
            if cur is not None:
                removed_pyc.append(path)
            continue
        tampered.append(f"{'deleted' if cur is None else 'modified'}: {path}")
    for path in have:
        if path in want:
            continue
        (removed_pyc if _is_bytecode(path) else removed_new).append(path)
    for udir, existed in man.get("user_sites", {}).items():
        if not existed and os.path.exists(udir):
            removed_new.append(udir)
    if not tampered:
        for path in removed_pyc + removed_new:
            try:
                if os.path.isdir(path) and not os.path.islink(path):
                    shutil.rmtree(path)
                else:
                    os.unlink(path)
            except OSError as exc:
                tampered.append(f"could not remove {path}: {exc}")
    lines = [
        f"site-check: {len(want)} build-time files under {len(man.get('dirs', []))} dir(s), "
        f"{len(man.get('interpreters', {}))} interpreter(s)",
        f"removed {len(removed_new)} file(s) added after build, {len(removed_pyc)} stale/new bytecode file(s)",
    ]
    lines += [f"  added: {p}" for p in removed_new[:50]]
    if tampered:
        lines.append(f"TAMPERED ({len(tampered)}): the build-time Python environment was modified")
        lines += [f"  {t}" for t in tampered[:50]]
    report = "\n".join(lines) + "\n"
    Path(args.out).write_text(report)
    sys.stdout.write(report)
    return 10 if tampered else 0


# --------------------------------------------------------------------------- #
# Build-free site-packages integrity check (pip RECORD + conda-meta).
#
# The agent phase is root, so it can edit installed dependency code in
# site-packages (numpy/__init__.py, river/<x>.py, a recorded .so) and plant
# sitecustomize.py / *.pth. The eval (uid 65534) and -- for the ~1/8 of tasks
# whose parser.py / score_spec.py import a third-party package -- the ROOT
# score step then import that tampered code and score on it. `-S` only stops
# site HOOKS in the root process; a normal `import numpy` still loads a
# tampered numpy, and the uid-65534 eval runs without -S at all.
#
# `site-check` closes this but needs tests/meta/site_manifest.json baked from a
# BUILT image (scripts/harbor/bake_interpreter_manifest.py) -- absent from the
# shipped bundles. `record-check` needs NO bake: the authoritative hashes
# already ship inside every image. pip writes `<dist>.dist-info/RECORD` (a
# sha256 per installed file); conda writes `conda-meta/<pkg>.json` with
# `paths_data` (sha256 / sha256_in_prefix per file). We refuse the run if any
# IMPORT-EXECUTABLE recorded file (.py/.so/.pyd/.pth) no longer matches ANY of
# its recorded hashes (pip and conda may both list one path post-layering), and
# we remove unrecorded sitecustomize.py / usercustomize.py / new top-level
# modules and unrecorded executable *.pth (minus the handful of legitimately
# unrecorded conda/setuptools .pth). Files neither installer records (bytecode,
# in-place-built .so, generated data) are left alone -- the conservative side.
#
# Scope: the interpreter's site-packages dirs (where RECORD / conda-meta live).
# The stdlib is not pip/conda-recorded, so a trojaned stdlib module is OUT of
# scope here (only the baked site_manifest catches that); documented residual.
# --------------------------------------------------------------------------- #

_RC_EXEC_SUFFIX = (".py", ".so", ".pyd", ".pth")
# Unrecorded *.pth that legitimately ship via conda/setuptools post-link scripts
# (observed in every pristine pytorch/conda image); never removed.
_RC_PTH_ALLOW = ("distutils-precedence.pth", "easy-install.pth",
                 "protobuf.pth", "__editable__.pth")


def _rc_is_exec(path: str) -> bool:
    return path.endswith(_RC_EXEC_SUFFIX) or ".so." in os.path.basename(path)


def _rc_site_dirs() -> list:
    import sysconfig
    dirs: list = []
    for k in ("purelib", "platlib"):
        p = sysconfig.get_path(k)
        if p and p not in dirs and os.path.isdir(p):
            dirs.append(p)
    try:
        import site as _site
        for p in _site.getsitepackages():
            if p and p not in dirs and os.path.isdir(p):
                dirs.append(p)
    except Exception:
        pass
    return dirs


def _rc_b64_to_hex(spec: str):
    if not spec or not spec.startswith("sha256="):
        return None
    import base64
    b = spec[7:]
    # Debian/Ubuntu dpkg-owned dist-info (apt python3-pip etc.) records the
    # digest as 64 hex chars, not PEP 376 urlsafe-b64 (43 chars) -- accept both.
    if len(b) == 64 and all(c in "0123456789abcdefABCDEF" for c in b):
        return b.lower()
    try:
        return base64.urlsafe_b64decode(b + "=" * (-len(b) % 4)).hex()
    except Exception:
        return None


def _rc_build_recorded(site_dirs: list, conda_prefixes: list) -> dict:
    """{abspath: set(sha256-hex)}; an empty set means recorded, hash unknown."""
    import glob
    rec: dict = {}

    def add(ap, h):
        s = rec.setdefault(ap, set())
        if h:
            s.add(h)

    for sp in site_dirs:
        for r in glob.glob(os.path.join(sp, "*.dist-info", "RECORD")):
            base = os.path.dirname(os.path.dirname(r))
            try:
                rows = open(r, encoding="utf-8").read().splitlines()
            except Exception:
                continue
            for row in rows:
                parts = row.rsplit(",", 2)
                if len(parts) != 3:
                    continue
                ap = os.path.normpath(os.path.join(base, parts[0]))
                add(ap, _rc_b64_to_hex(parts[1]))
        for r in glob.glob(os.path.join(sp, "*.egg-info", "installed-files.txt")):
            base = os.path.dirname(r)
            try:
                rows = open(r, encoding="utf-8").read().splitlines()
            except Exception:
                continue
            for rel in rows:
                add(os.path.normpath(os.path.join(base, rel.strip())), None)
    # dpkg is an installer that records too: apt `python3-*` packages put
    # top-level modules (six.py, pyparsing.py, ...) into /usr/lib/python3/
    # dist-packages with a `<pkg>.md5sums` list but no RECORD, so without this
    # an image whose verifier python is the distro python3 would have them
    # REMOVED as "new top-level modules" and its imports broken.
    roots = tuple(os.path.join(sp, "") for sp in site_dirs)
    for ms in glob.glob("/var/lib/dpkg/info/*.md5sums"):
        try:
            rows = open(ms, encoding="utf-8", errors="replace").read().splitlines()
        except Exception:
            continue
        for row in rows:
            parts = row.split(None, 1)
            if len(parts) != 2 or len(parts[0]) != 32:
                continue
            ap = os.path.normpath("/" + parts[1].strip())
            if ap.startswith(roots):
                add(ap, "md5:" + parts[0].lower())
    for pref in conda_prefixes:
        for j in glob.glob(os.path.join(pref, "conda-meta", "*.json")):
            try:
                d = json.load(open(j))
            except Exception:
                continue
            paths = d.get("paths_data", {}).get("paths", [])
            # `files` holds the REAL install-relative paths; for `noarch: python`
            # packages paths_data._path is a placeholder ("site-packages/..",
            # "python-scripts/..") that conda remaps at link time, so prefer
            # `files` (index-aligned with paths) for the path and paths[i] for
            # the hash. Fall back to _path when files is absent/mismatched.
            files = d.get("files") or []
            use_files = len(files) == len(paths) and files
            for i, pd in enumerate(paths):
                rel = files[i] if use_files else pd.get("_path")
                if not rel:
                    continue
                ap = os.path.normpath(os.path.join(pref, rel))
                add(ap, pd.get("sha256"))
                add(ap, pd.get("sha256_in_prefix"))
            if not use_files:
                for rel in files:
                    add(os.path.normpath(os.path.join(pref, rel)), None)
    return rec


def _rc_conda_prefixes(site_dirs: list) -> list:
    out: list = []
    for sp in site_dirs:
        p = sp
        for _ in range(7):
            if os.path.isdir(os.path.join(p, "conda-meta")):
                out.append(p)
                break
            nxt = os.path.dirname(p)
            if nxt == p:
                break
            p = nxt
    for c in ("/opt/conda", "/opt/miniconda3", sys.prefix):
        if os.path.isdir(os.path.join(c, "conda-meta")):
            out.append(c)
    return sorted(set(out))


def _rc_pth_executes(path: str) -> bool:
    try:
        for ln in open(path, encoding="utf-8").read().splitlines():
            s = ln.strip()
            if s.startswith(("import ", "import\t", "exec(", "exec ")):
                return True
    except Exception:
        return True  # unreadable -> treat as suspicious
    return False


def _rc_load_allow(path) -> dict:
    """tests/meta/record_check_allow.json: what the IMAGE BUILD itself did to
    the interpreter tree, measured on a clean build at render time.

      "modified": {abspath: [sha256, ...]}  a recorded file the Dockerfile
                  patched after installing it (sed/patch/python write). Its
                  live hash must equal one listed; [] = path-only entry (the
                  build could not be measured or is not byte-reproducible).
      "keep":     {abspath: [sha256, ...]}  an unrecorded top-level module /
                  executable .pth the build legitimately installed (setup.py,
                  apt, a hand-dropped shim); never removed. A listed hash that
                  no longer matches is a modified build-time file (refused);
                  [] = path-only.

    The file lives under /tests (verifier-only), so the agent can neither read
    nor extend it. A missing file is an empty allowlist."""
    if not path or not os.path.isfile(path):
        return {"modified": {}, "keep": {}}
    d = json.loads(Path(path).read_text())
    return {k: {p: [h.lower() for h in (v or [])] for p, v in (d.get(k) or {}).items()}
            for k in ("modified", "keep")}


def cmd_record_check(args: argparse.Namespace) -> int:
    allow = _rc_load_allow(getattr(args, "allow", None))
    site_dirs = _rc_site_dirs()
    conda_prefixes = _rc_conda_prefixes(site_dirs)
    rec = _rc_build_recorded(site_dirs, conda_prefixes)
    mismatches: list = []
    removable: list = []
    allowed: list = []
    emit: dict = {"modified": {}, "keep": {}}
    checked = 0

    def _allowed(kind: str, ap: str, live) -> bool:
        if ap not in allow[kind]:
            return False
        hs = allow[kind][ap]
        return not hs or (live is not None and live in hs)

    for sp in site_dirs:
        for dp, _dn, fns in os.walk(sp):
            for fn in fns:
                ap = os.path.join(dp, fn)
                if os.path.islink(ap):
                    continue
                if ap in rec:
                    hs = rec[ap]
                    if hs and _rc_is_exec(ap):
                        checked += 1
                        try:
                            live = _hash_file(ap)
                            if live not in hs and any(h.startswith("md5:") for h in hs):
                                with open(ap, "rb") as fh:
                                    if "md5:" + hashlib.md5(fh.read()).hexdigest() in hs:
                                        continue
                        except OSError:
                            live = None
                        if live is not None and live in hs:
                            continue
                        emit["modified"][ap] = [live] if live else []
                        if _allowed("modified", ap, live):
                            allowed.append(ap)
                        else:
                            mismatches.append(ap if live else ap + " (unreadable)")
                    continue
                if _is_bytecode(ap):
                    continue
                rel = os.path.relpath(ap, sp)
                top = os.sep not in rel
                if fn in ("sitecustomize.py", "usercustomize.py"):
                    cand = True
                elif fn.endswith(".pth"):
                    cand = fn not in _RC_PTH_ALLOW and _rc_pth_executes(ap)
                else:
                    cand = top and fn.endswith(".py")
                if not cand:
                    continue
                try:
                    live = _hash_file(ap)
                except OSError:
                    live = None
                emit["keep"][ap] = [live] if live else []
                if ap in allow["keep"]:
                    if _allowed("keep", ap, live):
                        allowed.append(ap)
                    else:
                        mismatches.append(ap + " (build-time unrecorded file changed)")
                    continue
                removable.append(ap)
    lines = [
        f"record-check: {len(rec)} recorded path(s) across {len(site_dirs)} site dir(s), "
        f"{len(conda_prefixes)} conda prefix(es); verified {checked} executable file(s)",
        f"allowlisted build-time changes present: {len(allowed)}",
        f"removable (unrecorded hooks / new top-level): {len(removable)}",
    ]
    lines += [f"  remove: {p}" for p in removable[:50]]
    if mismatches:
        lines.append(f"TAMPERED ({len(mismatches)}): a recorded dependency file was modified")
        lines += [f"  modified: {p}" for p in mismatches[:50]]
    report = "\n".join(lines) + "\n"
    if args.out and args.out != "-":
        Path(args.out).write_text(report)
    sys.stdout.write(report)
    if getattr(args, "emit_allow", None):
        # What THIS interpreter tree would need allowlisted to pass: run on a
        # clean build (no agent phase) at render time -> tests/meta allowlist.
        Path(args.emit_allow).write_text(json.dumps(
            {"version": 1, "modified": emit["modified"], "keep": emit["keep"]},
            indent=1, sort_keys=True) + "\n")
    if args.mode == "enforce":
        if mismatches:
            return 10
        for p in removable:
            try:
                os.unlink(p)
            except OSError:
                pass
    return 0



# --------------------------------------------------------------------------- #
# Shared hardened launcher for TRUSTED root python in eval scripts
#
#   $MLSBENCH_TRUSTED_PYTHON script.py args...     (or  -  /  -c CODE  /  -m MOD)
#
# expands to `<verifier python> -I -S -B /tests/score_task.py trusted-run --`.
# The target then runs with: no site hooks (sitecustomize, usercustomize,
# *.pth code), no user site, no PYTHON* env, no cwd or script directory on
# sys.path, no bytecode writes -- and the image's genuine site-packages
# appended explicitly (plus the plain directory lines of its *.pth files, so
# editable installs still import; no .pth line is executed). Under a site
# manifest (see site-check) those site-packages were verified before the evals.
# This is the ALFWorld / mlvision-lab pattern made reusable; a keep-root task's
# stager, oracle or scorer should start its root python through it.
# --------------------------------------------------------------------------- #

def _trusted_site_paths() -> list:
    import sysconfig
    paths: list = []
    for key in ("purelib", "platlib"):
        p = sysconfig.get_path(key)
        if p and p not in paths:
            paths.append(p)
    try:
        import site as _site
        for p in _site.getsitepackages():
            if p and p not in paths:
                paths.append(p)
    except Exception:
        pass
    extra: list = []
    for d in list(paths):
        try:
            names = sorted(os.listdir(d))
        except OSError:
            continue
        for n in names:
            if not n.endswith(".pth"):
                continue
            try:
                lines = open(os.path.join(d, n)).read().splitlines()
            except OSError:
                continue
            for line in lines:
                line = line.strip()
                if not line or line.startswith(("#", "import ", "import\t")):
                    continue
                cand = line if os.path.isabs(line) else os.path.join(d, line)
                if os.path.isdir(cand) and cand not in paths and cand not in extra:
                    extra.append(cand)
    return paths + extra


def trusted_run(argv: list) -> int:
    import runpy
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path
                   if p and os.path.abspath(p) != here and os.path.abspath(p) != os.getcwd()]
    sys.path.extend(p for p in _trusted_site_paths() if p not in sys.path)
    if not argv:
        print("usage: trusted-run -- <script|-|-c CODE|-m MOD> [args...]", file=sys.stderr)
        return 2
    target, rest = argv[0], argv[1:]
    if target == "-c" and rest:
        sys.argv = ["-c"] + rest[1:]
        exec(compile(rest[0], "<trusted -c>", "exec"), {"__name__": "__main__"})
    elif target == "-m" and rest:
        sys.argv = [rest[0]] + rest[1:]
        runpy.run_module(rest[0], run_name="__main__", alter_sys=True)
    elif target == "-":
        sys.argv = ["-"] + rest
        exec(compile(sys.stdin.read(), "<stdin>", "exec"), {"__name__": "__main__"})
    else:
        sys.argv = [target] + rest
        runpy.run_path(target, run_name="__main__")
    return 0


# --------------------------------------------------------------------------- #
# run-untrusted: the shared untrusted-phase runner for keep-root evals
#
#   $MLSBENCH_RUN_UNTRUSTED [--writable DIR]... [--strip REGEX]... [--lock NAME]
#                           [--gid N] [--env K=V]... [--clean-env] [--pass-fd N]...
#                           [--stdio relay|inherit] [--timeout SEC] -- CMD ARGS...
#
# Unifies what chip-macro-placement-search, fm-probability-path,
# llm-early-exit-criterion, security-machine-unlearning and rl-offline-adroit
# each hand-rolled (setpriv + per-run uid + pkill -u + find -uid -delete +
# ipcrm). Must be started as root (refuses otherwise: a task that opted into a
# split must never run its untrusted phase with the trusted uid). Per call:
#   * a fresh uid from 40000-49999, reserved in /run/mlsb-untrusted (0700) and
#     never reused in this container, so nothing it leaves can be touched by a
#     later phase that happens to get the same uid; gid = uid unless --gid;
#     no supplementary groups; PR_SET_NO_NEW_PRIVS;
#   * its own HOME / TMPDIR / XDG_CACHE_HOME (0700, owned by the uid) and the
#     eval env minus MLSBENCH_RESULT_FILE / MLSBENCH_RUN_UNTRUSTED; stdin is
#     /dev/null and every other descriptor is closed;
#   * its stdout+stderr go to a pipe THIS process drains and relays line by
#     line (minus lines matching any --strip regex) -- the untrusted uid never
#     holds the verifier's stdout, so once this returns nothing it left behind
#     can write there;
#   * when CMD exits: SIGKILL every process of the uid (kill(-1) as the uid,
#     repeated until /proc shows none), final drain, then --writable dirs are
#     chowned back to root with group/other write bits removed (what the
#     trusted phase reads next is immutable to every other uid), files the uid
#     owns under /tmp /var/tmp /dev/shm /run/lock /dev/mqueue are deleted and
#     its SysV IPC objects removed (as the owner: root lacks CAP_SYS_ADMIN);
#   * --lock NAME holds a container-wide flock across run + sweep (serialises
#     e.g. the seeds of one setting when they could pool a metered budget);
#   * --stdio inherit hands CMD this process's own stdin/stdout/stderr instead
#     of the relay -- for a trusted harness that talks to its untrusted worker
#     over the worker's stdio (Popen(helper..., stdin=PIPE, stdout=PIPE)); never
#     use it where fd 1 is the eval's stdout. --pass-fd N keeps descriptor N open
#     in CMD (an IPC pipe passed by number). --clean-env starts CMD from an
#     empty environment (PATH, LANG, HOME, TMPDIR, XDG_CACHE_HOME + --env);
#   * SIGTERM / SIGINT / SIGHUP to the helper kill the uid and still sweep
#     (exit 143 etc.); a SIGKILLed helper sweeps nothing and records nothing,
#     so a harness ends its worker by closing its stdin or terminate(), and
#     falls back to kill() only after a grace period;
#   * exit status: CMD's (128+signal when killed), 124 on --timeout, 125 if a
#     process of the uid survived SIGKILL (treat as a failed run).
# Natively the variable is unset and `$MLSBENCH_RUN_UNTRUSTED cmd` is just `cmd`.
# --------------------------------------------------------------------------- #

_UNTRUSTED_UID_BASE = 40000
_UNTRUSTED_UID_COUNT = 10000
_UNTRUSTED_LEDGER = "/run/mlsb-untrusted"
_SHARED_TMP_ROOTS = ("/tmp", "/var/tmp", "/dev/shm", "/run/lock", "/dev/mqueue")
# --writable footgun guard: never hand these trees (or these exact roots) to the
# untrusted uid. SAVE_PATH/OUTPUT_DIR live under /logs/verifier/save, which is
# the one part of /logs a task may open.
_UNTRUSTED_DENY = ("/proc", "/sys", "/dev", "/etc", "/usr", "/bin", "/sbin", "/lib",
                   "/lib32", "/lib64", "/libx32", "/boot", "/tests", "/solution",
                   "/logs", _UNTRUSTED_LEDGER)
_UNTRUSTED_DENY_EXACT = ("/", "/workspace", "/root", "/tmp", "/var", "/opt", "/data", "/run")
_UNTRUSTED_ALLOW_UNDER = ("/logs/verifier/save/",)


def _writable_refused(path: str) -> bool:
    if path in _UNTRUSTED_DENY_EXACT:
        return True
    if any(path.startswith(a) for a in _UNTRUSTED_ALLOW_UNDER):
        return False
    return any(path == d or path.startswith(d + "/") for d in _UNTRUSTED_DENY)


def _untrusted_ledger() -> Path:
    led = Path(_UNTRUSTED_LEDGER)
    try:
        os.mkdir(led, 0o700)
    except FileExistsError:
        pass
    st = os.lstat(led)
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != 0 or st.st_mode & 0o077:
        raise RuntimeError(f"{led} is not a private root directory")
    return led


_PSEUDO_FS = {"proc", "sysfs", "cgroup", "cgroup2", "devpts", "devtmpfs", "mqueue",
              "tracefs", "debugfs", "securityfs", "pstore", "bpf", "fusectl",
              "configfs", "hugetlbfs", "autofs", "binfmt_misc", "nsfs", "efivarfs"}


def _dormant_untrusted_uids(ledger: Path) -> set[int]:
    """uids in the pool that already own a file somewhere in the container
    (a tarball layer extracted with its archive's numeric owner, a baked
    dataset, a site-packages file): a phase running as such a uid could modify
    that file. Computed once per container -- one walk of every real mount --
    and cached in the ledger; the pool skips them."""
    import fcntl
    cache = ledger / "dormant-uids.json"
    with open(ledger / ".dormant.lock", "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if cache.is_file():
            try:
                return set(json.loads(cache.read_text())["uids"])
            except (OSError, ValueError, KeyError):
                pass
        lo, hi = _UNTRUSTED_UID_BASE, _UNTRUSTED_UID_BASE + _UNTRUSTED_UID_COUNT
        found: set[int] = set()
        roots: list[str] = []
        try:
            for line in Path("/proc/mounts").read_text().splitlines():
                parts = line.split()
                if len(parts) >= 3 and parts[2] not in _PSEUDO_FS:
                    roots.append(parts[1].replace("\\040", " "))
        except OSError:
            roots = ["/"]
        t0 = time.time()
        seen_dirs: set = set()
        stack = sorted(set(roots))
        n = 0
        while stack:
            d = stack.pop()
            try:
                dst = os.lstat(d)
            except OSError:
                continue
            key = (dst.st_dev, dst.st_ino)
            if key in seen_dirs or str(d).startswith(("/proc", "/sys")):
                continue
            seen_dirs.add(key)
            if lo <= dst.st_uid < hi:
                found.add(dst.st_uid)
            try:
                it = os.scandir(d)
            except OSError:
                continue
            with it:
                for ent in it:
                    try:
                        st = ent.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    n += 1
                    if lo <= st.st_uid < hi:
                        found.add(st.st_uid)
                    if stat.S_ISDIR(st.st_mode):
                        stack.append(ent.path)
        tmp = cache.with_suffix(".tmp")
        tmp.write_text(json.dumps({"uids": sorted(found), "entries": n,
                                   "seconds": round(time.time() - t0, 2)}))
        os.replace(tmp, cache)
        return found


def _alloc_untrusted_uid(ledger: Path) -> int:
    busy: set[int] = set(_dormant_untrusted_uids(ledger))
    try:
        for pid in (x for x in os.listdir("/proc") if x.isdigit()):
            try:
                st = os.stat(f"/proc/{pid}")
                busy.add(st.st_uid)
            except OSError:
                pass
    except OSError:
        pass
    for uid in range(_UNTRUSTED_UID_BASE, _UNTRUSTED_UID_BASE + _UNTRUSTED_UID_COUNT):
        if uid in busy:
            continue
        try:
            fd = os.open(ledger / str(uid), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            continue
        os.write(fd, f"{os.getpid()} {time.time():.3f}\n".encode())
        os.close(fd)
        return uid
    raise RuntimeError("untrusted uid pool exhausted")


def _chown_tree_nofollow(root: Path, uid: int, gid: int, strip_go_w: bool = False) -> None:
    def one(path: str) -> None:
        try:
            os.lchown(path, uid, gid)
            if strip_go_w:
                st = os.lstat(path)
                if not stat.S_ISLNK(st.st_mode) and st.st_mode & 0o022:
                    os.chmod(path, stat.S_IMODE(st.st_mode) & ~0o022)
        except OSError:
            pass
    one(str(root))
    for dirpath, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            one(os.path.join(dirpath, name))


def _sweep_uid_files(uid: int) -> int:
    removed = 0
    for root in _SHARED_TMP_ROOTS:
        try:
            rst = os.lstat(root)
        except OSError:
            continue
        if not stat.S_ISDIR(rst.st_mode):
            continue
        for dirpath, dirs, files in os.walk(root, topdown=False, followlinks=False):
            for name in files + dirs:
                path = os.path.join(dirpath, name)
                try:
                    st = os.lstat(path)
                except OSError:
                    continue
                if st.st_dev != rst.st_dev or st.st_uid != uid:
                    continue
                try:
                    if stat.S_ISDIR(st.st_mode):
                        shutil.rmtree(path)
                    else:
                        os.unlink(path)
                    removed += 1
                except OSError:
                    pass
    return removed


_IPC_RM_SNIPPET = (
    "import ctypes, json, sys\n"
    "libc = ctypes.CDLL(None)\n"
    "for kind, ident in json.loads(sys.argv[1]):\n"
    "    if kind == 'shm': libc.shmctl(ident, 0, None)\n"
    "    elif kind == 'msg': libc.msgctl(ident, 0, None)\n"
    "    elif kind == 'sem': libc.semctl(ident, 0, 0)\n"
)


def _remove_uid_ipc(uid: int, gid: int) -> int:
    ids: list = []
    for kind in ("shm", "msg", "sem"):
        try:
            lines = Path(f"/proc/sysvipc/{kind}").read_text().splitlines()
        except OSError:
            continue
        if not lines:
            continue
        head = lines[0].split()
        try:
            c_uid, c_id = head.index("uid"), head.index({"shm": "shmid", "msg": "msqid", "sem": "semid"}[kind])
        except ValueError:
            continue
        for line in lines[1:]:
            cols = line.split()
            if len(cols) > max(c_uid, c_id) and cols[c_uid] == str(uid):
                ids.append([kind, int(cols[c_id])])
    if ids:
        try:
            subprocess.run([sys.executable, "-I", "-S", "-c", _IPC_RM_SNIPPET, json.dumps(ids)],
                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=30,
                           **_drop_privileges_kwargs(uid, gid))
        except Exception:
            pass
    return len(ids)


def cmd_run_untrusted(argv: list) -> int:
    ap = argparse.ArgumentParser(prog="run-untrusted")
    ap.add_argument("--writable", action="append", default=[])
    ap.add_argument("--strip", action="append", default=[])
    ap.add_argument("--lock", default=None)
    ap.add_argument("--gid", type=int, default=None)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--timeout", type=float, default=0.0)
    ap.add_argument("--clean-env", action="store_true")
    ap.add_argument("--pass-fd", type=int, action="append", default=[])
    ap.add_argument("--stdio", choices=("relay", "inherit"), default="relay")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    say = lambda msg: print(f"[run-untrusted] {msg}", file=sys.stderr, flush=True)  # noqa: E731
    if not cmd:
        say("usage: run-untrusted [options] -- CMD ARGS...")
        return 2
    if os.geteuid() != 0:
        say(f"refused: started as uid {os.geteuid()}, not root -- the untrusted phase "
            "cannot be split from the trusted one")
        return 126
    strip = [re.compile(s) for s in a.strip]
    writable: list[Path] = []
    for w in a.writable:
        wp = Path(os.path.abspath(w))
        if _writable_refused(str(wp)):
            say(f"refused --writable {w}: a system or verifier path")
            return 2
        if wp.is_symlink():
            say(f"refused --writable {w}: a symlink")
            return 2
        wp.mkdir(parents=True, exist_ok=True)
        writable.append(wp)
    for fd in a.pass_fd:
        try:
            if fd < 3:
                raise OSError("0-2 are the stdio")
            os.fstat(fd)
        except OSError as exc:
            say(f"bad --pass-fd {fd}: {exc}")
            return 2
    extra_env = {}
    for kv in a.env:
        k, sep, v = kv.partition("=")
        if not sep or not k:
            say(f"bad --env {kv!r}")
            return 2
        extra_env[k] = v

    ledger = _untrusted_ledger()
    lock_fh = None
    if a.lock:
        import fcntl
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", a.lock)
        lock_fh = open(ledger / f"lock-{name}", "a")
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
    uid = _alloc_untrusted_uid(ledger)
    gid = uid if a.gid is None else int(a.gid)
    base = Path(tempfile.mkdtemp(prefix=f"mlsb-untrusted-{uid}-", dir="/tmp"))
    home, tmp = base / "home", base / "tmp"
    home.mkdir()
    tmp.mkdir()
    for d in (base, home, tmp):
        os.chown(d, uid, gid)
        os.chmod(d, 0o700)
    for wp in writable:
        _chown_tree_nofollow(wp, uid, gid)

    result_file = os.environ.get("MLSBENCH_RESULT_FILE", "")
    if a.clean_env:
        env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8"}
    else:
        env = dict(os.environ)
    for k in ("MLSBENCH_RESULT_FILE", "MLSBENCH_RUN_UNTRUSTED"):
        env.pop(k, None)
    env.update({"HOME": str(home), "TMPDIR": str(tmp), "XDG_CACHE_HOME": str(home / ".cache")})
    env.update(extra_env)

    import ctypes
    try:
        libc = ctypes.CDLL(None, use_errno=True)
    except OSError:
        libc = None

    def demote() -> None:  # pragma: no cover - runs in the forked child
        os.setgroups([])
        os.setgid(gid)
        os.setuid(uid)
        if libc is not None:
            libc.prctl(38, 1, 0, 0, 0)  # PR_SET_NO_NEW_PRIVS

    rfd, wfd = os.pipe()
    wakeup = _ChildWakeup()
    buf = b""

    def relay(chunk: bytes, final: bool = False) -> None:
        nonlocal buf
        buf += chunk
        lines = buf.split(b"\n")
        buf = lines.pop()
        if final and buf:
            lines.append(buf)
            buf = b""
        out = [ln + b"\n" for ln in lines
               if not any(r.search(ln.decode("utf-8", "replace")) for r in strip)]
        if out:
            try:
                os.write(1, b"".join(out))
            except OSError:
                pass

    def drain() -> None:
        while True:
            try:
                chunk = os.read(rfd, 1 << 16)
            except (BlockingIOError, OSError):
                return
            if not chunk:
                return
            relay(chunk)

    stop = {"sig": 0}

    def on_signal(signum, _frame) -> None:
        stop["sig"] = signum

    old_sigs = {}
    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            old_sigs[s] = signal.signal(s, on_signal)
        except (ValueError, OSError):
            pass

    inherit = a.stdio == "inherit"
    rc = 0
    survivors = -1
    try:
        try:
            if inherit:
                io_kw = {}
            else:
                io_kw = {"stdin": subprocess.DEVNULL, "stdout": wfd, "stderr": wfd}
            proc = subprocess.Popen(cmd, env=env, close_fds=True, pass_fds=tuple(a.pass_fd),
                                    start_new_session=True, preexec_fn=demote, **io_kw)
        except OSError as exc:
            os.close(wfd)
            say(f"cannot start {cmd[0]!r}: {exc}")
            rc = 127
            proc = None
        if proc is not None:
            os.close(wfd)
            os.set_blocking(rfd, False)
            deadline = time.time() + a.timeout if a.timeout > 0 else None
            timed_out = False
            while proc.poll() is None and not stop["sig"]:
                if deadline is not None and time.time() > deadline:
                    timed_out = True
                    break
                watch = ([] if inherit else [rfd]) + ([wakeup.fd] if wakeup.fd is not None else [])
                try:
                    ready, _, _ = select.select(watch, [], [], 0.1)
                except (OSError, ValueError):
                    ready = []
                if wakeup.fd is not None and wakeup.fd in ready:
                    wakeup.drain()
                if rfd in ready:
                    drain()
            left = _kill_uid_everywhere(uid, gid)
            survivors = left
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
            drain()
            relay(b"", final=True)
            if stop["sig"]:
                say(f"stopped by signal {stop['sig']}")
                rc = 128 + stop["sig"]
            elif timed_out:
                say(f"timed out after {a.timeout:g}s")
                rc = 124
            elif left:
                say(f"{left} process(es) of uid {uid} survived SIGKILL")
                rc = 125
            else:
                rc = proc.returncode if proc.returncode >= 0 else 128 - proc.returncode
    finally:
        wakeup.close()
        try:
            os.close(rfd)
        except OSError:
            pass
        if _uid_pids(uid):
            _kill_uid_everywhere(uid, gid)
        for wp in writable:
            _chown_tree_nofollow(wp, 0, 0, strip_go_w=True)
        _remove_uid_ipc(uid, gid)
        _sweep_uid_files(uid)
        shutil.rmtree(base, ignore_errors=True)
        if lock_fh is not None:
            lock_fh.close()
        _record_untrusted_run(result_file, uid, rc, survivors, cmd)
        for s, h in old_sigs.items():
            try:
                signal.signal(s, h)
            except (ValueError, OSError):
                pass
    return rc


def _record_untrusted_run(result_file: str, uid: int, rc: int, survivors: int, cmd: list) -> None:
    """Append this invocation to the (label, seed)'s sidecar, which `score`
    requires for a harbor_trusted_result task. Only when the result file sits
    in a private root directory (the verifier's)."""
    if not result_file:
        return
    rp = Path(result_file)
    try:
        dst = os.lstat(rp.parent)
        if not stat.S_ISDIR(dst.st_mode) or dst.st_uid != 0 or dst.st_mode & 0o077:
            return
        fd = os.open(_untrusted_sidecar(rp), os.O_WRONLY | os.O_CREAT | os.O_APPEND
                     | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            prog = os.path.basename(str(cmd[0])) if cmd else "?"
            os.write(fd, f"uid={uid} rc={rc} survivors={survivors} cmd={prog}\n".encode())
        finally:
            os.close(fd)
    except OSError:
        pass


def main(argv: list[str] | None = None) -> int:
    raw = sys.argv[1:] if argv is None else list(argv)
    if raw and raw[0] == "run-untrusted":
        return cmd_run_untrusted(raw[1:])
    if raw and raw[0] == "trusted-run":
        rest = raw[1:]
        if rest and rest[0] == "--":
            rest = rest[1:]
        return trusted_run(rest)
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("guard")
    g.add_argument("--task-meta", required=True)
    g.add_argument("--pristine", required=True, help="Workdir-level pristine root, e.g. /opt/mlsbench/original")
    g.add_argument("--workspace", required=True, help="Workdir-level workspace root, e.g. /workspace")
    g.add_argument("--violation-out", required=True)
    g.set_defaults(func=cmd_guard)

    r = sub.add_parser("run-evals")
    r.add_argument("--task-meta", required=True)
    r.add_argument("--workspace", required=True, help="Workdir-level workspace root, e.g. /workspace")
    r.add_argument("--eval-root", required=True, help="Dir containing scripts/ — e.g. /tests/eval")
    r.add_argument("--out-dir", required=True)
    r.add_argument("--oracle-cmd-overrides", default=None,
                   help="Oracle-only JSON list of {label, cmd} substitutions.")
    r.set_defaults(func=cmd_run_evals)

    s = sub.add_parser("score")
    s.add_argument("--task-meta", required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--reward-out", required=True)
    s.set_defaults(func=cmd_score)

    m = sub.add_parser("site-manifest")
    m.add_argument("--out", default="-")
    m.add_argument("--workspace", default="/workspace")
    m.set_defaults(func=cmd_site_manifest)

    c = sub.add_parser("site-check")
    c.add_argument("--manifest", required=True)
    c.add_argument("--out", required=True)
    c.set_defaults(func=cmd_site_check)

    rc = sub.add_parser("record-check")
    rc.add_argument("--mode", default="check", choices=["check", "enforce"])
    rc.add_argument("--out", default="-")
    rc.add_argument("--allow", default=None,
                    help="tests/meta/record_check_allow.json (build-time changes to accept)")
    rc.add_argument("--emit-allow", default=None,
                    help="write the allowlist this tree would need (run on a clean build)")
    rc.set_defaults(func=cmd_record_check)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
