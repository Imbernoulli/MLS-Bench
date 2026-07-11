"""Core adapter logic: read MLS-Bench task tree, emit Harbor task dirs."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import stat
import statistics
import sys
import tempfile
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import tomli_w
from jinja2 import Environment, FileSystemLoader, StrictUndefined


# --------------------------------------------------------------------------- #
# MLS-Bench paths
# --------------------------------------------------------------------------- #

TEMPLATE_DIR = Path(__file__).parent / "task-template"


@dataclass
class MlsBenchRoot:
    """Resolved MLS-Bench checkout.

    Two on-disk layouts are supported for the vendored package source:
      - `vendor/<pkg>/`                 — what `mlsbench fetch` writes by
                                          default from a fresh clone
      - `vendor/external_packages/<pkg>/` — what the maintainer's experiment
                                          checkout uses, with prepared data
                                          under `vendor/data/<dep>/` and
                                          per-package Dockerfiles under
                                          `vendor/images/`

    `package_src(pkg)` picks whichever exists.
    """
    root: Path

    @property
    def tasks_dir(self) -> Path: return self.root / "tasks"
    @property
    def vendor_dir(self) -> Path: return self.root / "vendor"
    @property
    def pkg_configs_dir(self) -> Path: return self.root / "vendor" / "pkg_configs"
    @property
    def src_dir(self) -> Path: return self.root / "src"

    def package_src(self, pkg: str) -> Path | None:
        for cand in (
            self.root / "vendor" / "external_packages" / pkg,
            self.root / "vendor" / pkg,
        ):
            if cand.exists():
                return cand
        norm = _normalize_pkg_name(pkg)
        for parent in (
            self.root / "vendor" / "external_packages",
            self.root / "vendor",
        ):
            if not parent.is_dir():
                continue
            for cand in parent.iterdir():
                if cand.is_dir() and _normalize_pkg_name(cand.name) == norm:
                    return cand
        return None

    # Back-compat for callers that want a directory (returns the parent that
    # actually contains the packages — `external_packages/` if that layout is
    # used, else `vendor/`).
    @property
    def vendor_pkgs_dir(self) -> Path:
        if (self.root / "vendor" / "external_packages").is_dir():
            return self.root / "vendor" / "external_packages"
        return self.root / "vendor"

    def list_tasks(self) -> list[str]:
        return sorted(p.name for p in self.tasks_dir.iterdir()
                      if p.is_dir() and (p / "config.json").exists())


def detect_mls_bench_root(explicit: Path | None) -> MlsBenchRoot:
    """Resolve an MLS-Bench checkout.

    Priority:
      1. --mls-bench-root if given
      2. $MLS_BENCH_ROOT env var
      3. nearest ancestor of cwd containing tasks/ + vendor/packages.yaml
      4. ~/MLS-Bench/ if it looks like a checkout (the maintainer's prepared
         experiment dir holds all vendored packages and prepared data under
         vendor/external_packages/ and vendor/data/, which is convenient for
         building harbor base images without re-fetching)
    """
    if explicit:
        return MlsBenchRoot(explicit.resolve())
    import os
    env = os.environ.get("MLS_BENCH_ROOT")
    if env:
        return MlsBenchRoot(Path(env).resolve())
    cur = Path.cwd().resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "tasks").is_dir() and (cand / "vendor" / "packages.yaml").is_file():
            return MlsBenchRoot(cand)
    home_default = Path.home() / "MLS-Bench"
    if (home_default / "tasks").is_dir() and (home_default / "vendor" / "packages.yaml").is_file():
        return MlsBenchRoot(home_default.resolve())
    raise FileNotFoundError(
        "Could not auto-detect an MLS-Bench checkout. "
        "Pass --mls-bench-root, set MLS_BENCH_ROOT, or run from inside a checkout."
    )


# --------------------------------------------------------------------------- #
# Per-task context assembly
# --------------------------------------------------------------------------- #

@dataclass
class TaskContext:
    task_id: str
    task_description: str
    config: dict
    package: str
    pkg_config: dict
    leaderboard_rows: list[dict] = field(default_factory=list)
    chosen_baseline: str = ""
    baseline_edit_ops: list[dict] = field(default_factory=list)


@dataclass
class AdapterRunResult:
    generated: int
    requested: int
    failed: list[tuple[str, Exception]] = field(default_factory=list)
    task_dirs: list[Path] = field(default_factory=list)


class MlsBenchAdapter:
    """Harbor adapter wrapper matching the standard adapter template."""

    def __init__(
        self,
        output_dir: Path,
        limit: int | None = None,
        overwrite: bool = False,
        task_ids: list[str] | None = None,
        mls_bench_root: Path | None = None,
        continue_on_error: bool = False,
        mangrove: bool = False,
        gpu_backend: str = "h20",
    ):
        self.output_dir = output_dir
        self.limit = limit
        self.overwrite = overwrite
        self.task_ids = task_ids
        self.mls_bench_root = mls_bench_root
        self.continue_on_error = continue_on_error
        self.mangrove = mangrove
        self.gpu_backend = gpu_backend

    def _wanted_task_ids(self, mb: MlsBenchRoot) -> list[str]:
        wanted = list(self.task_ids) if self.task_ids else mb.list_tasks()
        if self.limit is not None:
            wanted = wanted[: self.limit]
        return wanted

    def run(self) -> AdapterRunResult:
        mb = detect_mls_bench_root(self.mls_bench_root)
        wanted = self._wanted_task_ids(mb)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        ok = 0
        failed: list[tuple[str, Exception]] = []
        task_dirs: list[Path] = []
        for task_id in wanted:
            try:
                ctx = build_task_context(mb, task_id)
                out = render_task(mb, ctx, self.output_dir, overwrite=self.overwrite, mangrove=self.mangrove, gpu_backend=self.gpu_backend)
                print(f"[ok] {task_id} -> {out}")
                ok += 1
                task_dirs.append(out)
            except Exception as exc:
                failed.append((task_id, exc))
                print(f"[fail] {task_id}: {exc}", file=sys.stderr)
                if not self.continue_on_error:
                    raise

        write_dataset_manifest(self.output_dir)
        return AdapterRunResult(ok, len(wanted), failed, task_dirs)


_TIME_RE = re.compile(r"^(\d+):(\d+):(\d+)$")

_ALLOWED_OP_IMPORT_ROOTS = {"custom_template", "importlib", "json", "math", "pathlib", "sys"}


def _warn(message: str) -> None:
    """Emit a warning that is visible in adapter logs."""
    warnings.warn(message, RuntimeWarning, stacklevel=2)
    print(f"[warn] {message}", file=sys.stderr)


def _normalize_pkg_name(name: str) -> str:
    """Match native package resolution: case/separator-insensitive."""
    return str(name).lower().replace("-", "").replace("_", "")


def _same_package_name(left: str, right: str) -> bool:
    return _normalize_pkg_name(left) == _normalize_pkg_name(right)


def _limited_import(
    name: str,
    globals: dict[str, Any] | None = None,
    locals: dict[str, Any] | None = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
) -> Any:
    root = name.split(".", 1)[0]
    if level != 0 or root not in _ALLOWED_OP_IMPORT_ROOTS:
        raise ImportError(f"import of {name!r} is not allowed while loading edit ops")
    return __import__(name, globals, locals, fromlist, level)


def _safe_open(file: str | os.PathLike, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
    if any(flag in mode for flag in ("w", "a", "x", "+")):
        raise ValueError("edit op files may only open files read-only")
    return open(file, mode, *args, **kwargs)


_SAFE_OP_BUILTINS: dict[str, Any] = {
    "__import__": _limited_import,
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "format": format,
    "frozenset": frozenset,
    "getattr": getattr,
    "hasattr": hasattr,
    "int": int,
    "iter": iter,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "next": next,
    "open": _safe_open,
    "print": print,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "setattr": setattr,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "zip": zip,
    "chr": chr,
    "divmod": divmod,
    "ord": ord,
    "AssertionError": AssertionError,
    "Exception": Exception,
    "ImportError": ImportError,
    "RuntimeError": RuntimeError,
    "ValueError": ValueError,
}


def _safe_rel_path(rel: str, *, field: str = "path") -> Path:
    if not isinstance(rel, str) or not rel.strip():
        raise ValueError(f"{field} must be a non-empty relative path")
    if "\\" in rel:
        raise ValueError(f"{field} must use POSIX separators: {rel!r}")
    p = Path(rel)
    if p.is_absolute() or any(part == ".." for part in p.parts):
        raise ValueError(f"{field} escapes the workspace: {rel!r}")
    if not p.parts or any(part in {"", "."} for part in p.parts):
        raise ValueError(f"{field} is not a normal relative path: {rel!r}")
    return p


def _safe_join(base: Path, rel: str, *, field: str = "path") -> Path:
    child = (base / _safe_rel_path(rel, field=field)).resolve()
    root = base.resolve()
    try:
        child.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field} escapes {root}: {rel!r}") from exc
    return child


def _safe_rel_str(rel: str, *, field: str = "path") -> str:
    return _safe_rel_path(rel, field=field).as_posix()


def _adapter_data_root(mb: MlsBenchRoot) -> Path:
    configured = os.environ.get("MLSBENCH_DATA_ROOT")
    return Path(configured).expanduser() if configured else mb.vendor_dir / "data"


def _expand_adapter_path_template(
    mb: MlsBenchRoot,
    task_dir: Path,
    value: str,
) -> Path:
    expanded = (
        str(value)
        .replace("{project_root}", str(mb.root))
        .replace("{data_root}", str(_adapter_data_root(mb)))
        .replace("{task_dir}", str(task_dir))
    )
    path = Path(expanded).expanduser()
    return path if path.is_absolute() else mb.root / path


def _absolute_prune_paths(config: dict, *, key: str, root: str) -> list[str]:
    raw_paths = config.get(key) or []
    if not isinstance(raw_paths, list):
        raise ValueError(f"{key} must be a list of absolute paths under {root}/")

    out: list[str] = []
    for idx, raw in enumerate(raw_paths):
        value = str(raw).strip()
        if any(char in value for char in "\r\n\0"):
            raise ValueError(f"{key}[{idx}] contains a control character")
        path = PurePosixPath(value)
        if (
            not path.is_absolute()
            or not value.startswith(f"{root}/")
            or any(part in {"", ".", ".."} for part in path.parts[1:])
        ):
            raise ValueError(
                f"{key}[{idx}] must be a normal absolute path below {root}/: {raw!r}"
            )
        out.append(shlex.quote(value))
    return out


def _agent_data_prune_paths(config: dict) -> list[str]:
    return _absolute_prune_paths(config, key="agent_data_prune", root="/data")


def _agent_image_prune_paths(config: dict) -> list[str]:
    return _absolute_prune_paths(config, key="agent_image_prune", root="/opt")


def _configured_package_paths(config: dict, *, key: str) -> set[str]:
    raw_paths = config.get(key) or []
    if not isinstance(raw_paths, list):
        raise ValueError(f"{key} must be a list of workspace-relative package paths")

    packages = _task_packages(config)
    out: set[str] = set()
    for idx, raw in enumerate(raw_paths):
        rel = _safe_rel_str(str(raw).strip(), field=f"{key}[{idx}]")
        parts = PurePosixPath(rel).parts
        if len(parts) < 2 or not any(_same_package_name(parts[0], pkg) for pkg in packages):
            raise ValueError(
                f"{key}[{idx}] must name a path below a configured test package: {raw!r}"
            )
        out.add(rel)
    return out


def _verifier_only_package_paths(config: dict) -> set[str]:
    return _configured_package_paths(config, key="verifier_only_package_files")


def _agent_pruned_package_paths(config: dict) -> set[str]:
    return _configured_package_paths(config, key="agent_pruned_package_files")


def _path_is_private(rel: str, private_paths: set[str]) -> bool:
    rel_path = PurePosixPath(rel)
    for private in private_paths:
        private_path = PurePosixPath(private)
        if rel_path == private_path or private_path in rel_path.parents:
            return True
    return False


def _privacy_opted_in(ctx: TaskContext, config: dict) -> bool:
    return bool(
        config.get("verifier_only_package_files")
        or config.get("agent_pruned_package_files")
        or ctx.pkg_config.get("agent_pruned_files")
    )


def _config_with_private_package_paths(
    mb: MlsBenchRoot,
    ctx: TaskContext,
    config: dict,
) -> dict:
    """Return a config whose private package paths are explicit and complete.

    Privacy filtering is opt-in. Once a task requests it, package-level prune
    entries, reference directories, and unrelated sibling solution files are
    removed from the agent scaffold. Verifier-only files are restored from the
    private tests mount immediately before evaluation; agent-pruned files are
    never restored.
    """
    out = deepcopy(config)
    if not _privacy_opted_in(ctx, out):
        return out

    configured = list(out.get("agent_pruned_package_files") or [])
    package_pruned = ctx.pkg_config.get("agent_pruned_files") or []
    if not isinstance(package_pruned, list):
        raise ValueError(
            f"package {ctx.package!r} agent_pruned_files must be a list"
        )
    for idx, raw in enumerate(package_pruned):
        subpath = _safe_rel_str(
            str(raw).strip(), field=f"package agent_pruned_files[{idx}]"
        )
        configured.append(f"{ctx.package}/{subpath}")

    out["agent_pruned_package_files"] = configured
    pruned = _agent_pruned_package_paths(out)
    public = {
        _safe_rel_str(str(entry["filename"]), field="files[].filename")
        for entry in out.get("files", [])
        if entry.get("filename")
    }
    public_entries = out.get("agent_public_package_files") or []
    if not isinstance(public_entries, list):
        raise ValueError("agent_public_package_files must be a list")
    for idx, raw in enumerate(public_entries):
        public.add(
            _safe_rel_str(str(raw), field=f"agent_public_package_files[{idx}]")
        )

    for package in _task_packages(out):
        package_src = _materialized_package_source(mb, package)
        if package_src is None:
            raise FileNotFoundError(
                f"privacy-filtered package source {package!r} is unavailable"
            )
        root = package_src.resolve()
        for path in package_src.rglob("*"):
            try:
                file_stat = path.lstat()
            except OSError:
                continue
            if not stat.S_ISREG(file_stat.st_mode):
                continue
            try:
                path.resolve().relative_to(root)
            except ValueError:
                continue
            subpath = path.relative_to(package_src)
            if any(part in {".git", "__pycache__"} for part in subpath.parts):
                continue
            if path.suffix in {".pyc", ".pyo"}:
                continue

            rel = f"{package}/{subpath.as_posix()}"
            is_reference = any(part in {"anchors", "baselines"} for part in subpath.parts)
            is_sibling_solution = (
                bool(subpath.parts)
                and subpath.parts[0] == "solution"
                and rel not in public
            )
            if is_reference or is_sibling_solution:
                pruned.add(rel)

    out["agent_pruned_package_files"] = sorted(pruned)
    return out


def _stage_verifier_data_deps(
    mb: MlsBenchRoot,
    task_dir: Path,
    meta: Path,
    config: dict,
) -> None:
    deps = config.get("verifier_data_deps") or []
    if not isinstance(deps, list):
        raise ValueError("verifier_data_deps must be a list")

    for idx, dep in enumerate(deps):
        if not isinstance(dep, dict):
            raise ValueError(f"verifier_data_deps[{idx}] must be an object")
        host_path = dep.get("host_path")
        destination = dep.get("dest")
        if not host_path or not destination:
            raise ValueError(
                f"verifier_data_deps[{idx}] must define host_path and dest"
            )
        source = _expand_adapter_path_template(mb, task_dir, str(host_path))
        dest = _safe_join(
            meta, str(destination), field=f"verifier_data_deps[{idx}].dest"
        )
        if not source.exists():
            message = (
                f"verifier data dependency {dep.get('name', idx)!r} is missing at "
                f"{source}; prepare it in the pinned image before rendering"
            )
            if bool(dep.get("required", True)):
                raise FileNotFoundError(message)
            _warn(message)
            continue
        if source.is_dir():
            shutil.copytree(
                source,
                dest,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
            )
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)


def _stage_verifier_only_package_files(
    mb: MlsBenchRoot,
    ctx: TaskContext,
    meta: Path,
    config: dict,
) -> None:
    verifier_paths = _verifier_only_package_paths(config)
    if not verifier_paths:
        return

    # Re-materialize the post-mid-edit workspace without removing the files
    # that must be private. This preserves task-specific harness edits instead
    # of silently restoring a stale copy from the package source.
    staging_config = deepcopy(config)
    staging_config["verifier_only_package_files"] = []
    with tempfile.TemporaryDirectory(prefix="mlsbench-verifier-package-") as temp:
        staged_root = Path(temp)
        _stage_task_scaffold(
            mb,
            ctx,
            staged_root,
            config=staging_config,
            force_clean_scaffold=True,
        )
        destination_root = meta / "verifier_package_files"
        for rel in sorted(verifier_paths):
            source = _safe_join(staged_root, rel, field="verifier_only_package_files")
            if not source.exists():
                raise FileNotFoundError(
                    f"verifier-only package path {rel!r} is absent from the "
                    "task-materialized workspace"
                )
            destination = _safe_join(
                destination_root, rel, field="verifier_only_package_files"
            )
            if source.is_dir():
                shutil.copytree(
                    source,
                    destination,
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
                )
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)


def _op_file_matches(op: dict, target_file: str) -> bool:
    try:
        return _safe_rel_path(op.get("file", ""), field="op.file") == _safe_rel_path(
            target_file, field="target_file"
        )
    except ValueError:
        return False


def _line_count(content: str) -> int:
    if not content.endswith("\n"):
        content += "\n"
    return len(content.splitlines())


def _content_lines(content: str) -> list[str]:
    if not content.endswith("\n"):
        content += "\n"
    return content.splitlines(keepends=True)


def _end_index(lines: list[str], end_line: int) -> int:
    return len(lines) if int(end_line) == -1 else int(end_line)


def _delete_bounds(op: dict) -> tuple[int, int]:
    if "start_line" in op:
        start = int(op["start_line"])
        end = int(op.get("end_line", start))
        return start, end
    line = int(op["line"])
    return line, line


def _subpath_under_package(rel: str, pkg: str) -> Path:
    parts = _safe_rel_path(rel, field="package file").parts
    if parts and _same_package_name(parts[0], pkg):
        return Path(*parts[1:]) if len(parts) > 1 else Path()
    return Path(*parts)


def _package_from_rel(config: dict, default_pkg: str, rel: str) -> str:
    parts = _safe_rel_path(rel, field="package file").parts
    if parts:
        first = parts[0]
        for pkg in _task_packages(config):
            if _same_package_name(first, pkg):
                return pkg
    return default_pkg


def _require_int(op: dict, key: str, source: Path, idx: int, *, minimum: int) -> int:
    try:
        value = int(op[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{source}: OPS[{idx}].{key} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"{source}: OPS[{idx}].{key} must be >= {minimum}")
    return value


def _validate_ops(raw_ops: Any, source: Path) -> list[dict]:
    if raw_ops is None:
        return []
    if not isinstance(raw_ops, list):
        raise ValueError(f"{source}: OPS must be a list of operation dictionaries")

    out: list[dict] = []
    for idx, raw in enumerate(raw_ops):
        if not isinstance(raw, dict):
            raise ValueError(f"{source}: OPS[{idx}] must be a dictionary")
        op = dict(raw)
        kind = op.get("op")
        if kind not in {"create", "replace", "insert", "delete"}:
            raise ValueError(f"{source}: OPS[{idx}].op has unsupported value {kind!r}")
        op["file"] = _safe_rel_str(op.get("file", ""), field=f"OPS[{idx}].file")

        if kind in {"create", "replace", "insert"} and not isinstance(op.get("content"), str):
            raise ValueError(f"{source}: OPS[{idx}].content must be a string")

        if kind == "replace":
            start = _require_int(op, "start_line", source, idx, minimum=1)
            try:
                end = int(op["end_line"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{source}: OPS[{idx}].end_line must be an integer") from exc
            if end != -1 and end < start:
                raise ValueError(f"{source}: OPS[{idx}].end_line must be >= start_line")
        elif kind == "insert":
            _require_int(op, "after_line", source, idx, minimum=0)
        elif kind == "delete":
            if "start_line" in op:
                start = _require_int(op, "start_line", source, idx, minimum=1)
                try:
                    end = int(op.get("end_line", start))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{source}: OPS[{idx}].end_line must be an integer") from exc
                if end != -1 and end < start:
                    raise ValueError(f"{source}: OPS[{idx}].end_line must be >= start_line")
            elif "line" in op:
                _require_int(op, "line", source, idx, minimum=1)
            else:
                raise ValueError(
                    f"{source}: OPS[{idx}] delete must declare start_line/end_line or line"
                )
        out.append(op)
    return out


def _load_ops_file(ops_py: Path) -> list[dict]:
    """Load an MLS-Bench OPS file in reduced globals and validate its schema.

    MLS-Bench represents scaffolding and baselines as Python snippets. This is
    not a security sandbox, but the adapter only needs the resulting ``OPS``:
    builtins are restricted, imports are limited to modules used for declarative
    op construction, and the returned operations must match the expected schema.
    """
    ns: dict[str, Any] = {
        "__builtins__": _SAFE_OP_BUILTINS,
        "__file__": str(ops_py),
        "__name__": "__mlsbench_ops__",
    }
    old_sys_path = list(sys.path)
    sentinel = object()
    saved_custom_template = sys.modules.pop("custom_template", sentinel)
    try:
        exec(compile(ops_py.read_text(), str(ops_py), "exec"), ns, ns)
        return _validate_ops(ns.get("OPS"), ops_py)
    finally:
        sys.path[:] = old_sys_path
        if saved_custom_template is sentinel:
            sys.modules.pop("custom_template", None)
        else:
            sys.modules["custom_template"] = saved_custom_template


def _assert_python_syntax(
    text: str,
    *,
    rel_path: str,
    origin: str,
    task_id: str,
) -> None:
    """Raise RuntimeError if ``text`` (the contents of ``rel_path`` after an op
    sequence) is not valid Python.

    The error message names the workspace-relative path, the originating op
    source (``origin`` — e.g. ``"mid_edit.py"`` or
    ``"pre_edit.py for Time-Series-Library"``), the failing line, and a small
    code snippet. This catches silent vendor-drift bugs where a
    hard-coded line-range op cuts a multi-line statement: pre_edit/mid_edit
    line numbers depend on the upstream source layout, and if a vendored
    package is re-fetched off the pinned commit the resulting scaffold would
    otherwise ship broken Python that only fails at agent runtime.
    """
    if not rel_path.endswith(".py"):
        return
    try:
        ast.parse(text, filename=rel_path)
    except SyntaxError as exc:
        snippet_lines = text.splitlines()
        lo = max(1, (exc.lineno or 1) - 2)
        hi = min(len(snippet_lines), (exc.lineno or 1) + 2)
        snippet = "\n".join(
            f"{'>>' if i == exc.lineno else '  '}{i:5d}: {snippet_lines[i-1]}"
            for i in range(lo, hi + 1)
        )
        raise RuntimeError(
            f"{task_id}: {origin} produced invalid Python in {rel_path!r} at "
            f"line {exc.lineno} ({exc.msg}). Most likely the vendored package "
            f"source has drifted from the line numbers the op file targets — "
            f"re-pin the upstream commit in vendor/packages.yaml or convert "
            f"the op to a content-anchored edit.\n{snippet}"
        ) from exc


def _parse_time(t: str) -> int:
    """`H:MM:SS` → seconds."""
    m = _TIME_RE.match(t.strip())
    if not m:
        return 3600
    h, mi, s = (int(x) for x in m.groups())
    return h * 3600 + mi * 60 + s


def _resolve_package(config: dict) -> str:
    packages = _task_packages(config)
    if packages:
        return packages[0]
    # Fallback: first editable file's top-level dir.
    for f in config.get("files", []):
        rel = _safe_rel_path(str(f["filename"]), field="files[].filename")
        return str(rel.parts[0])
    return ""


def _task_packages(config: dict) -> list[str]:
    packages: list[str] = []
    seen: set[str] = set()
    for tc in config.get("test_cmds", []):
        pkg = tc.get("package")
        if not pkg:
            continue
        pkg = str(pkg)
        norm = _normalize_pkg_name(pkg)
        if norm in seen:
            continue
        seen.add(norm)
        packages.append(pkg)
    return packages


def _load_pkg_config(mb: MlsBenchRoot, pkg: str) -> dict:
    cfg_path = mb.pkg_configs_dir / pkg / "config.json"
    if not cfg_path.exists():
        norm = _normalize_pkg_name(pkg)
        if mb.pkg_configs_dir.is_dir():
            for cand in mb.pkg_configs_dir.iterdir():
                if cand.is_dir() and _normalize_pkg_name(cand.name) == norm:
                    cfg_path = cand / "config.json"
                    break
    if not cfg_path.exists():
        return {}
    return json.loads(cfg_path.read_text())


def _load_leaderboard(task_dir: Path) -> list[dict]:
    lb = task_dir / "leaderboard.csv"
    if not lb.exists():
        return []
    with lb.open() as f:
        return list(csv.DictReader(f))


def _pick_strongest_baseline(
    task_dir: Path,
    config: dict,
    leaderboard: list[dict],
) -> str:
    """Pick the baseline whose leaderboard row scores highest under the task's
    native ``score_spec.py``.

    Delegates entirely to ``mlsbench.scoring``: each baseline row is scored
    with ``score_record`` against ``BaselineAnchors``, exactly like a real
    agent submission. This avoids any ad-hoc label-vs-column string matching
    in the adapter.

    Falls back to the first declared baseline only when scoring is unavailable
    (no ``score_spec.py``, no leaderboard rows, or import failure).
    """
    baselines = config.get("baselines") or {}
    if not baselines:
        return ""
    first = next(iter(baselines))

    if not leaderboard:
        return first

    try:
        from mlsbench.scoring.anchors import BaselineAnchors
        from mlsbench.scoring.evaluate import load_expanded_spec, score_record
    except Exception as exc:
        _warn(
            f"_pick_strongest_baseline: mlsbench.scoring not importable "
            f"({exc!r}); falling back to first declared baseline {first!r} "
            f"for {task_dir.name}."
        )
        return first

    anchors = BaselineAnchors(task_dir)
    spec = load_expanded_spec(task_dir, anchors)
    if spec is None or not spec.settings:
        _warn(
            f"_pick_strongest_baseline: no usable score_spec for "
            f"{task_dir.name}; falling back to first declared baseline "
            f"{first!r}."
        )
        return first

    # For each baseline, find its leaderboard row(s) and score them with the
    # native scorer. Selection precedence matches `evaluate.evaluate_task`:
    # prefer is_final + seed=mean, then seed=mean, then any row, picking the
    # most metric-complete row within the chosen tier.
    def _completeness(rec: dict) -> int:
        c = 0
        for k, v in rec.items():
            if k in {"timestamp", "model", "is_final", "seed"}:
                continue
            if k.startswith("elapsed_") or k.endswith("_std"):
                continue
            if v in ("", None):
                continue
            if isinstance(v, str) and v.strip().lower() in {"nan", "null", "none"}:
                continue
            try:
                f = float(v)
            except (TypeError, ValueError):
                c += 1
                continue
            if not math.isnan(f):
                c += 1
        return c

    def _pick_row(rows: list[dict]) -> dict | None:
        if not rows:
            return None
        return max(rows, key=lambda r: (_completeness(r), str(r.get("timestamp", ""))))

    # Only consider baselines that have edit_ops; baselines without edit_ops
    # (e.g. built-in models in Time-Series-Library) can't be applied as oracle
    # edits and may exceed parameter budgets.
    eligible = {n for n in baselines if baselines[n].get("edit_ops")}
    if not eligible:
        eligible = set(baselines)  # fall back to all if none have edit_ops

    best_name: str | None = None
    best_score: float = float("-inf")
    any_scored = False
    for name in baselines:
        if name not in eligible:
            continue
        candidates = {name, f"baseline:{name}"}
        rows = [r for r in leaderboard if r.get("model") in candidates]
        if not rows:
            continue
        final_mean = [r for r in rows
                      if r.get("seed") == "mean"
                      and str(r.get("is_final", "")).lower() == "true"]
        mean_rows = [r for r in rows if r.get("seed") == "mean"]
        chosen = _pick_row(final_mean) or _pick_row(mean_rows) or _pick_row(rows)
        if chosen is None:
            continue
        try:
            s = score_record(spec, chosen, anchors)
        except Exception as exc:
            _warn(
                f"_pick_strongest_baseline: scoring baseline {name!r} for "
                f"{task_dir.name} raised {exc!r}; skipping."
            )
            continue
        if s is None or (isinstance(s, float) and math.isnan(s)):
            continue
        any_scored = True
        if s > best_score:
            best_name, best_score = name, float(s)

    if best_name is None:
        if not any_scored:
            _warn(
                f"_pick_strongest_baseline: no baseline row could be scored "
                f"for {task_dir.name}; falling back to first declared "
                f"baseline {first!r}."
            )
        return first
    return best_name


def build_task_context(mb: MlsBenchRoot, task_id: str) -> TaskContext:
    task_dir = mb.tasks_dir / task_id
    config = json.loads((task_dir / "config.json").read_text())
    task_description = (task_dir / "task_description.md").read_text()
    package = _resolve_package(config)
    pkg_config = _load_pkg_config(mb, package)
    leaderboard = _load_leaderboard(task_dir)

    # Pick baseline + load its edit ops (resolved relative to task_dir).
    # Ensure mlsbench.scoring is importable so _pick_strongest_baseline can
    # score baselines properly (instead of falling back to first-declared).
    _src_str = str(mb.src_dir)
    if _src_str not in sys.path:
        sys.path.insert(0, _src_str)
    baseline_name = ""
    edit_ops: list[dict] = []
    baselines = config.get("baselines") or {}
    if baselines:
        baseline_name = _pick_strongest_baseline(task_dir, config, leaderboard)
        rel = baselines[baseline_name].get("edit_ops")
        if rel:
            try:
                ops_py = _safe_join(task_dir, rel, field="baseline edit_ops")
                edit_ops = _load_ops_file(ops_py)
            except Exception as exc:
                raise RuntimeError(
                    f"{task_id}: could not load oracle baseline ops {rel}: {exc}"
                ) from exc

    return TaskContext(
        task_id=task_id,
        task_description=task_description,
        config=config,
        package=package,
        pkg_config=pkg_config,
        leaderboard_rows=leaderboard,
        chosen_baseline=baseline_name,
        baseline_edit_ops=edit_ops,
    )


# --------------------------------------------------------------------------- #
# Render
# --------------------------------------------------------------------------- #

PREBUILT_DOCKER = "bohanlyu2022/mlsbench-{pkg}:latest"
# The harbor-flavored image inherits from the prebuilt one and bakes in the
# pristine package source, mlsbench src, and any data deps. See
# harbor_adapter/scripts/build_base_image.py.
HARBOR_BASE_DOCKER = "bohanlyu2022/mlsbench-harbor-{pkg}:latest"
MANGROVE_CPU_ONLY_MIN_CPUS = 8
MANGROVE_CPU_ONLY_MEMORY_MB = 64 * 1024
MANGROVE_GPU_MIN_MEMORY_MB = 128 * 1024
MANGROVE_GPU_MEMORY_PER_GPU_MB = 96 * 1024
MANGROVE_CPUS_PER_GPU = 8
MANGROVE_CPU_STORAGE_MB = 40 * 1024
MANGROVE_GPU_STORAGE_MB = 80 * 1024
MANGROVE_MAX_GPUS_PER_TASK = 4
MANGROVE_AGENT_TIMEOUT_SEC = 1800
_PINNED_IMAGE_RE = re.compile(r"^[^\s]+@sha256:[0-9a-f]{64}$")


def _base_image(pkg: str, mangrove: bool, pkg_config: dict | None = None) -> str:
    """Return the FROM image for the per-task Dockerfile."""
    if not mangrove:
        return HARBOR_BASE_DOCKER.format(pkg=pkg)

    config = pkg_config or {}
    configured = {
        str(value).strip()
        for value in (
            config.get("mangrove_base_image"),
            config.get("pinned_harbor_image"),
        )
        if value
    }
    if not configured:
        raise ValueError(
            f"package {pkg!r} must define mangrove_base_image or "
            "pinned_harbor_image with an immutable @sha256 digest; refusing "
            "the mutable :latest fallback"
        )
    if len(configured) != 1:
        raise ValueError(
            f"package {pkg!r} defines conflicting pinned Mangrove images: "
            f"{sorted(configured)}"
        )
    image = configured.pop()
    if not _PINNED_IMAGE_RE.fullmatch(image):
        raise ValueError(
            f"package {pkg!r} Mangrove image must be pinned by @sha256: {image!r}"
        )
    return image


def _harbor_safe_name(task_id: str) -> str:
    safe = re.sub(r"[^a-z0-9_-]", "-", task_id.lower())
    return f"mls-bench__{safe}"


def _agent_timeout_sec(config: dict) -> int:
    # Every configured setting participates in assessment. Legacy disclosure
    # metadata must not alter the agent budget.
    eval_total = sum(
        _parse_time(tc.get("time", "0:30:00"))
        for tc in config.get("test_cmds", [])
    )
    # 30 min minimum, 0.5× of eval time as the heuristic, 4h hard cap.
    return max(30 * 60, min(4 * 3600, eval_total // 2 or 30 * 60))


def _group_test_cmds(test_cmds: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    auto_group = 10000
    for tc in test_cmds:
        group = tc.get("group")
        if group is None:
            group = auto_group
            auto_group += 1
        grouped.setdefault(group, []).append(tc)
    return grouped


def _test_cmd_compute(tc: dict) -> float:
    try:
        return float(tc.get("compute", 1) or 1)
    except (TypeError, ValueError):
        return 1.0


def _seed_count(config: dict) -> int:
    seeds = config.get("seeds") or [42]
    if isinstance(seeds, int):
        return 1
    try:
        return max(1, len(seeds))
    except TypeError:
        return 1


def _verifier_timeout_sec(config: dict, *, serial: bool = False) -> int:
    test_cmds = list(config.get("test_cmds", []) or [])
    n_seeds = _seed_count(config)
    if serial:
        total = sum(_parse_time(tc.get("time", "0:30:00")) for tc in test_cmds) * n_seeds
    else:
        total = sum(
            max(_parse_time(tc.get("time", "0:30:00")) for tc in entries)
            for entries in _group_test_cmds(test_cmds).values()
        )
    return total + 30 * 60 + 120 * len(test_cmds) * n_seeds


def _resources(pkg_config: dict, config: dict) -> dict:
    use_cuda = bool(config.get("use_cuda")) or bool(pkg_config.get("use_cuda"))
    cpus = 4
    memory_mb = 16 * 1024 if use_cuda else 8 * 1024
    storage_mb = 60 * 1024 if use_cuda else 30 * 1024
    gpus = 0
    if use_cuda:
        n_seeds = _seed_count(config)
        peak_gpus = 0
        for entries in _group_test_cmds(list(config.get("test_cmds", []) or [])).values():
            whole_per_seed = 0
            fractional_per_seed = 0.0
            for tc in entries:
                compute = _test_cmd_compute(tc)
                if compute >= 1.0:
                    whole_per_seed += max(1, math.ceil(compute))
                elif compute > 0.0:
                    fractional_per_seed += compute
            total_whole = n_seeds * whole_per_seed
            total_fractional = n_seeds * fractional_per_seed
            peak_gpus = max(
                peak_gpus,
                total_whole + max(0, math.ceil(total_fractional)),
            )
        gpus = max(1, peak_gpus)
    return dict(cpus=cpus, memory_mb=memory_mb, storage_mb=storage_mb, gpus=gpus)


def _mangrove_cpus(gpus: int) -> int:
    if gpus <= 0:
        return MANGROVE_CPU_ONLY_MIN_CPUS
    return max(MANGROVE_CPU_ONLY_MIN_CPUS, MANGROVE_CPUS_PER_GPU * gpus)


def _mangrove_memory_mb(gpus: int, requested_mem_mb: int) -> int:
    if gpus <= 0:
        return max(MANGROVE_CPU_ONLY_MEMORY_MB, requested_mem_mb)
    return max(
        MANGROVE_GPU_MIN_MEMORY_MB,
        MANGROVE_GPU_MEMORY_PER_GPU_MB * gpus,
        requested_mem_mb,
    )


def _mangrove_resource_overrides(pkg_config: dict, config: dict) -> dict[str, int]:
    allowed = {"cpus", "memory_mb", "storage_mb"}
    overrides: dict[str, int] = {}
    for owner, source in (("package", pkg_config), ("task", config)):
        raw = source.get("mangrove_resources") or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{owner} mangrove_resources must be an object")
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(
                f"{owner} mangrove_resources has unsupported field(s): {sorted(unknown)}"
            )
        for field, value in raw.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"{owner} mangrove_resources.{field} must be a positive integer"
                )
            if value <= 0:
                raise ValueError(
                    f"{owner} mangrove_resources.{field} must be a positive integer"
                )
            overrides[field] = value
    return overrides


def _apply_mangrove_resource_overrides(
    resources: dict,
    pkg_config: dict,
    config: dict,
) -> dict:
    out = dict(resources)
    out.update(_mangrove_resource_overrides(pkg_config, config))
    return out


def _mangrove_resources(
    pkg_config: dict,
    config: dict,
    gpu_backend: str = "h20",
) -> dict:
    """Compute GPU-targeted resource allocation for Mangrove.

    gpu_backend controls which hardware pool to target:
      - "b200": B200-MIG-1g.23gb slices for compute <= 1, full B200
        cards for compute ≥ 2. Requires torch 2.7.0 upgrade for sm_100.
      - "h20" (default): H20 whole cards (96 GB, SM 90). The pinned image
        must already contain the runtime; per-task verification never installs
        a replacement torch build.
    """
    if gpu_backend not in {"h20", "b200"}:
        raise ValueError(f"unsupported Mangrove GPU backend: {gpu_backend!r}")
    test_cmds = list(config.get("test_cmds", []) or [])
    max_compute = max((_test_cmd_compute(tc) for tc in test_cmds), default=0.0)
    requested_mem_mb = max(
        (int(math.ceil(float(tc.get("mem", 0) or 0))) * 1024 for tc in test_cmds),
        default=0,
    )

    # Explicit use_cuda=False → no GPU regardless of compute values.
    if config.get("use_cuda") is False or pkg_config.get("use_cuda") is False:
        return _apply_mangrove_resource_overrides(
            dict(
                cpus=_mangrove_cpus(0),
                memory_mb=_mangrove_memory_mb(0, requested_mem_mb),
                storage_mb=MANGROVE_CPU_STORAGE_MB,
                gpus=0, gpu_types=[], batch_size_multiplier=1,
            ),
            pkg_config,
            config,
        )

    # Check if all compute values are explicitly 0 (CPU-only tasks).
    all_zero = all(
        float(tc.get("compute", 0) or 0) <= 0
        for tc in test_cmds
    ) if test_cmds else True
    use_cuda = bool(config.get("use_cuda")) or bool(pkg_config.get("use_cuda"))

    if all_zero and not use_cuda:
        return _apply_mangrove_resource_overrides(
            dict(
                cpus=_mangrove_cpus(0),
                memory_mb=_mangrove_memory_mb(0, requested_mem_mb),
                storage_mb=MANGROVE_CPU_STORAGE_MB,
                gpus=0, gpu_types=[], batch_size_multiplier=1,
            ),
            pkg_config,
            config,
        )

    # ---- H20 backend ----
    if gpu_backend == "h20":
        serial = config.get("_verifier_serial") is True
        if serial:
            peak_gpus = max(1, math.ceil(max_compute)) if (use_cuda or not all_zero) else 0
        else:
            n_seeds = _seed_count(config)
            peak_gpus = 0
            for entries in _group_test_cmds(test_cmds).values():
                whole_per_seed = 0
                fractional_per_seed = 0.0
                for tc in entries:
                    compute = _test_cmd_compute(tc)
                    if compute >= 1.0:
                        whole_per_seed += max(1, math.ceil(compute))
                    elif compute > 0.0:
                        fractional_per_seed += compute
                peak_gpus = max(
                    peak_gpus,
                    n_seeds * whole_per_seed
                    + max(0, math.ceil(n_seeds * fractional_per_seed)),
                )
        gpus = max(1, peak_gpus) if (use_cuda or not all_zero) else 0
        if gpus > MANGROVE_MAX_GPUS_PER_TASK:
            raise ValueError(
                f"Mangrove task requires {gpus} H20 GPUs, above the "
                f"{MANGROVE_MAX_GPUS_PER_TASK}-GPU task cap; set "
                "_verifier_serial=true when settings can run in waves"
            )
        return _apply_mangrove_resource_overrides(
            dict(
                cpus=_mangrove_cpus(gpus),
                memory_mb=_mangrove_memory_mb(gpus, requested_mem_mb),
                storage_mb=MANGROVE_GPU_STORAGE_MB,
                gpus=gpus, gpu_types=["H20"], batch_size_multiplier=1,
            ),
            pkg_config,
            config,
        )

    # ---- B200 backend (default) ----
    # Tasks where individual cmds need ≥ 2 H100s use full B200 cards.
    # B200 ≈ 2× H100 (gpu_cap=2), so divide H100-equiv demand by 2.
    if max_compute >= 2.0:
        gpu_cap = 2
        n_seeds = _seed_count(config)
        peak_h100_equiv = 0
        for entries in _group_test_cmds(test_cmds).values():
            group_h100_equiv = 0
            for tc in entries:
                compute = _test_cmd_compute(tc)
                group_h100_equiv += max(1, math.ceil(compute))
            peak_h100_equiv = max(peak_h100_equiv, n_seeds * group_h100_equiv)
        gpus = max(1, math.ceil(peak_h100_equiv / gpu_cap))
        return _apply_mangrove_resource_overrides(
            dict(
                cpus=_mangrove_cpus(gpus),
                memory_mb=_mangrove_memory_mb(gpus, requested_mem_mb),
                storage_mb=MANGROVE_GPU_STORAGE_MB,
                gpus=gpus, gpu_types=["B200"], batch_size_multiplier=2,
                gpu_compute_cap=gpu_cap,
            ),
            pkg_config,
            config,
        )

    # All cmds have compute ≤ 1.0 → use MIG slices.
    # Each MIG slice runs exactly one experiment, so count the peak number
    # of parallel experiments (cmds × seeds) within any single group.
    n_seeds = _seed_count(config)
    peak_experiments = 0
    for entries in _group_test_cmds(test_cmds).values():
        # Every (cmd, seed) pair is one experiment needing one MIG slice.
        peak_experiments = max(peak_experiments, len(entries) * n_seeds)

    gpus = max(1, peak_experiments) if (peak_experiments or use_cuda) else 0

    return _apply_mangrove_resource_overrides(
        dict(
            cpus=_mangrove_cpus(gpus),
            memory_mb=_mangrove_memory_mb(gpus, requested_mem_mb),
            storage_mb=MANGROVE_GPU_STORAGE_MB,
            gpus=gpus, gpu_types=["B200-MIG-1g.23gb"], batch_size_multiplier=1,
        ),
        pkg_config,
        config,
    )


def _editable_files_view(config: dict) -> list[dict]:
    out = []
    for f in config.get("files", []):
        edits = f.get("edit") or []
        if not edits:
            continue
        filename = _safe_rel_str(str(f["filename"]), field="files[].filename")
        edit_full = any(int(r["start"]) == -1 and int(r["end"]) == -1 for r in edits)
        out.append({
            "filename": filename,
            "edit": [{"start": int(r["start"]), "end": int(r["end"])} for r in edits if not (int(r["start"]) == -1 and int(r["end"]) == -1)],
            "edit_full": edit_full,
        })
    return out


def _readable_only_files(config: dict) -> list[dict]:
    out = []
    for f in config.get("files", []):
        edits = f.get("edit") or []
        if edits: continue
        out.append({"filename": _safe_rel_str(str(f["filename"]), field="files[].filename")})
    return out


def _has_budget_check(task_dir: Path) -> bool:
    return (task_dir / "budget_check.py").exists()


def _budget_multiplier(task_dir: Path) -> float:
    p = task_dir / "budget_check.py"
    if not p.exists():
        return 1.05
    m = re.search(r"BUDGET_MULTIPLIER\s*=\s*([0-9.]+)", p.read_text())
    return float(m.group(1)) if m else 1.05


def render_task(
    mb: MlsBenchRoot,
    ctx: TaskContext,
    out_root: Path,
    overwrite: bool = False,
    mangrove: bool = False,
    gpu_backend: str = "h20",
) -> Path:
    if mangrove:
        out_dir = out_root / ctx.task_id
    else:
        out_dir = out_root / _harbor_safe_name(ctx.task_id)
    if out_dir.exists():
        if not overwrite:
            return out_dir
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    task_dir = mb.tasks_dir / ctx.task_id

    pkg_workdir = ctx.pkg_config.get("workdir", "/workspace")
    if mangrove:
        res = _mangrove_resources(ctx.pkg_config, ctx.config, gpu_backend=gpu_backend)
    else:
        res = _resources(ctx.pkg_config, ctx.config)
    effective_config = _config_with_shifted_edit_ranges(mb, ctx)
    effective_config = _config_with_private_package_paths(mb, ctx, effective_config)
    private_paths = (
        _verifier_only_package_paths(effective_config)
        | _agent_pruned_package_paths(effective_config)
    )
    clean_scaffold_packages = (
        sorted(
            set(
                _task_packages(effective_config)
                + ([ctx.package] if ctx.package else [])
            )
        )
        if private_paths
        else []
    )
    workdir_path = PurePosixPath(str(pkg_workdir))
    if not workdir_path.is_absolute() or any(
        part in {"", ".", ".."} for part in workdir_path.parts[1:]
    ):
        raise ValueError(f"package workdir must be a normal absolute path: {pkg_workdir!r}")
    clean_scaffold_paths: list[str] = []
    for package in clean_scaffold_packages:
        package_path = _safe_rel_path(package, field="package name")
        if len(package_path.parts) != 1:
            raise ValueError(f"package name must be one path component: {package!r}")
        clean_scaffold_paths.append(
            shlex.quote(str(workdir_path / package_path.as_posix()))
        )

    baseline_sections, baseline_warnings = _baseline_sections(mb, ctx, config=effective_config)
    read_sections, read_warnings = _read_sections(mb, ctx, config=effective_config)

    # Mangrove overrides: bounded agent timeout, no internet, H20 resources.
    agent_timeout = MANGROVE_AGENT_TIMEOUT_SEC if mangrove else _agent_timeout_sec(ctx.config)
    allow_internet = False if mangrove else True
    gpu_types = res.get("gpu_types", []) if mangrove else []
    batch_size_multiplier = res.get("batch_size_multiplier", 1) if mangrove else 1

    # Upgrade torch only for B200/B300 (sm_100 needs torch ≥ 2.7.0).
    # H20 is SM 90 (Hopper) — existing base-image torch works as-is.
    torch_upgrade = mangrove and gpu_backend != "h20"

    template_ctx = {
        "task_id": ctx.task_id,
        "task_description": ctx.task_description,
        "package": ctx.package,
        "workdir": pkg_workdir,
        "base_image": _base_image(ctx.package.lower(), mangrove, ctx.pkg_config),
        "pkg_base_image": ctx.pkg_config.get("base_image", "python:3.11"),
        "pkg_install_cmds": ctx.pkg_config.get("install_cmds") or [],
        "agent_image_prune_paths": _agent_image_prune_paths(effective_config),
        "agent_data_prune_paths": _agent_data_prune_paths(effective_config),
        "clean_scaffold_packages": clean_scaffold_packages,
        "clean_scaffold_paths": clean_scaffold_paths,
        "agent_timeout_sec": agent_timeout,
        "verifier_timeout_sec": _verifier_timeout_sec(
            ctx.config,
            serial=ctx.config.get("_verifier_serial") is True,
        ) * (3 if mangrove and gpu_backend != "h20" else 1),
        "build_timeout_sec": 3600,
        "cpus": res["cpus"],
        "memory_mb": res["memory_mb"],
        "storage_mb": res["storage_mb"],
        "gpus": res["gpus"],
        "gpu_types": gpu_types,
        "allow_internet": allow_internet,
        "mangrove": mangrove,
        "torch_upgrade": torch_upgrade,
        "batch_size_multiplier": batch_size_multiplier,
        "difficulty": ctx.config.get("difficulty", "hard"),
        "category": "ml-research",
        "tags": _domain_tags(ctx.task_id),
        "editable_files": _editable_files_view(effective_config),
        "extra_readable_files": _readable_only_files(effective_config),
        "has_budget_check": _has_budget_check(task_dir),
        "budget_multiplier": _budget_multiplier(task_dir),
        "baseline_name": ctx.chosen_baseline or "noop",
        "env_vars": ctx.pkg_config.get("env") or {},
        "data_deps": ctx.pkg_config.get("data_deps") or [],
        "gpu_count": res["gpus"],
        "baseline_sections": baseline_sections,
        "read_sections": read_sections,
        "prompt_warnings": baseline_warnings + read_warnings,
    }

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )

    # task.toml — uses [environment].docker_image so no Dockerfile is needed.
    (out_dir / "task.toml").write_text(env.get_template("task.toml.j2").render(**template_ctx))

    # instruction.md — agent's prompt; embeds all baseline editable regions.
    (out_dir / "instruction.md").write_text(env.get_template("instruction.md.j2").render(**template_ctx))

    # environment/ — tiny per-task layer FROM the harbor base. Carries any
    # mid_edit `create` scaffolding files the agent needs at workspace start.
    env_dir = out_dir / "environment"
    env_dir.mkdir()
    scaffold_files = _stage_task_scaffold(
        mb,
        ctx,
        env_dir / "_scaffold",
        config=effective_config,
    )
    (env_dir / "Dockerfile").write_text(
        env.get_template("environment/Dockerfile.j2").render(
            **template_ctx,
            scaffold_files=scaffold_files,
        )
    )

    # GPU reservation: Harbor's base docker-compose only sets cpu/memory
    # limits, not GPU. Tasks that resolve to `gpus > 0` need a per-task
    # compose override so docker actually attaches the nvidia runtime.
    # Compose-merge with base happens via harbor/environments/docker/
    # docker.py:292 picking up this file when present. `res["gpus"]`
    # comes from `_resources()` which honors both task config's `use_cuda`
    # and the package config's `use_cuda` flag.
    gpus_int = int(res.get("gpus") or 0)
    if gpus_int > 0 and not mangrove:
        (env_dir / "docker-compose.yaml").write_text(
            "services:\n"
            "  main:\n"
            "    deploy:\n"
            "      resources:\n"
            "        reservations:\n"
            "          devices:\n"
            "            - driver: nvidia\n"
            f"              count: {gpus_int}\n"
            "              capabilities: [gpu]\n"
        )

    # solution/ — Harbor mounts this at /solution/ only when the oracle agent runs.
    sol_dir = out_dir / "solution"
    sol_dir.mkdir()
    solve = sol_dir / "solve.sh"
    solve.write_text(env.get_template("solution/solve.sh.j2").render(**template_ctx))
    solve.chmod(solve.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    (sol_dir / "baseline_edit_ops.json").write_text(
        json.dumps(ctx.baseline_edit_ops, indent=2)
    )

    # tests/ — Harbor mounts this at /tests/ only during verification.
    tests_dir = out_dir / "tests"
    tests_dir.mkdir()
    test_sh = tests_dir / "test.sh"
    shutil.copy2(TEMPLATE_DIR / "tests" / "test.sh", test_sh)
    test_sh.chmod(test_sh.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    shutil.copy2(TEMPLATE_DIR / "tests" / "score_task.py", tests_dir / "score_task.py")

    # Per-task secrets that the agent must NOT see. Live under tests/ which
    # Harbor mounts only at verify time.
    _stage_verifier_assets(
        mb, ctx, tests_dir, task_dir,
        scaffold_dir=env_dir / "_scaffold",
        config=effective_config,
        mangrove=mangrove,
        gpu_backend=gpu_backend,
    )

    return out_dir


def _manual_task_digest(task_dir: Path) -> str:
    ignored_dirs = {"__pycache__"}
    ignored_suffixes = {".pyc", ".pyo"}
    files: list[Path] = []
    for rel in (
        "task.toml",
        "instruction.md",
        "README.md",
    ):
        p = task_dir / rel
        if p.exists():
            files.append(p)
    for rel_dir in ("environment", "tests", "solution", "steps"):
        root = task_dir / rel_dir
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if any(part in ignored_dirs for part in p.relative_to(task_dir).parts):
                continue
            if p.suffix in ignored_suffixes:
                continue
            files.append(p)
    files.sort(key=lambda p: p.relative_to(task_dir).as_posix())
    digest = hashlib.sha256()
    for p in files:
        rel = p.relative_to(task_dir).as_posix()
        file_hash = hashlib.sha256(p.read_bytes()).hexdigest()
        digest.update(f"{rel}\0{file_hash}\n".encode())
    return digest.hexdigest()


def _task_manifest_entry(task_dir: Path) -> dict:
    try:
        from harbor.models.task.config import TaskConfig
        from harbor.publisher.packager import Packager

        config = TaskConfig.model_validate_toml((task_dir / "task.toml").read_text())
        if config.task is None:
            raise ValueError(f"task.toml in {task_dir} has no [task] section")
        content_hash, _ = Packager.compute_content_hash(task_dir)
        return {"name": config.task.name, "digest": f"sha256:{content_hash}"}
    except Exception:
        text = (task_dir / "task.toml").read_text()
        match = re.search(r'(?m)^name\s*=\s*"([^"]+)"\s*$', text)
        if not match:
            raise
        return {"name": match.group(1), "digest": f"sha256:{_manual_task_digest(task_dir)}"}


def write_dataset_manifest(output_dir: Path) -> Path:
    """Write Harbor's dataset.toml manifest next to generated task dirs."""
    task_dirs = sorted(
        p for p in output_dir.iterdir()
        if p.is_dir() and (p / "task.toml").exists()
    )
    tasks = [_task_manifest_entry(task_dir) for task_dir in task_dirs]
    manifest = {
        "dataset": {
            "name": "mls-bench/mls-bench",
            "description": (
                "MLS-Bench Harbor adapter dataset containing the 140 "
                "algorithmic machine-learning research tasks."
            ),
            "authors": [{"name": "MLS-Bench authors", "email": "bohan22@stanford.edu"}],
            "keywords": ["ml-research", "algorithm-design", "multi-seed"],
        },
        "tasks": tasks,
    }
    path = output_dir / "dataset.toml"
    header = (
        "# Dataset manifest for mls-bench/mls-bench\n"
        "# Regenerate with: python -m mls_bench.main --output-dir <dir>\n\n"
    )
    path.write_text(header + tomli_w.dumps(manifest))
    return path


def _stage_task_scaffold(
    mb: MlsBenchRoot,
    ctx: TaskContext,
    scaffold_dst: Path,
    config: dict | None = None,
    force_clean_scaffold: bool = False,
) -> list[str]:
    """Stage the task's mid_edit ops as a tree of final files under
    `scaffold_dst/<workdir-relative-path>`.

    Handles all four op kinds:
      - create:  writes op.content verbatim
      - replace: reads the target file from vendor source (+ pre_edit applied
                 if a package-level pre_edit.py exists), replaces lines
                 start_line..end_line with op.content, writes the result
      - insert:  inserts op.content after op.after_line into the target
      - delete:  removes lines start_line..end_line from the target

    The per-task Dockerfile COPY overwrites the base image's version with
    our materialized version, equivalent to applying mid_edit in-place at
    runtime. Returns the list of workdir-relative paths produced.
    """
    task_dir = mb.tasks_dir / ctx.task_id
    mid_edit = task_dir / "edits" / "mid_edit.py"
    ops = _load_ops_file(mid_edit) if mid_edit.exists() else []
    config = config or ctx.config
    private_paths = (
        _verifier_only_package_paths(config)
        | _agent_pruned_package_paths(config)
    )
    clean_scaffold = bool(private_paths) or force_clean_scaffold

    def _remove_private_paths(package: str, destination: Path) -> None:
        for rel in sorted(private_paths):
            rel_path = PurePosixPath(rel)
            if not rel_path.parts or not _same_package_name(rel_path.parts[0], package):
                continue
            private_destination = destination / Path(*rel_path.parts[1:])
            if private_destination.is_dir() and not private_destination.is_symlink():
                shutil.rmtree(private_destination)
            elif private_destination.exists() or private_destination.is_symlink():
                private_destination.unlink()

    # Cache target-file text per rel path so multiple ops on the same file
    # apply in sequence (matches native BaseAgent.apply_mid_edit semantics).
    file_texts: dict[str, list[str]] = {}

    def _load_target(rel: str) -> list[str] | None:
        if rel in file_texts:
            return file_texts[rel]
        pkg = _package_from_rel(config, ctx.package, rel)
        pkg_src_path = _materialized_package_source(mb, pkg)
        if pkg_src_path is None:
            return None
        sub = _subpath_under_package(rel, pkg)
        src = pkg_src_path / sub
        if not src.exists():
            return None
        file_texts[rel] = src.read_text().splitlines(keepends=True)
        return file_texts[rel]

    created: list[str] = []

    packages = list(_task_packages(config))
    if ctx.package and not any(_same_package_name(ctx.package, pkg) for pkg in packages):
        packages.append(ctx.package)

    if clean_scaffold:
        # A privacy-filtered task cannot inherit an opaque package tree from a
        # lower image layer. Materialize the full reviewed package, remove all
        # private paths, and let the Dockerfile replace the old tree atomically.
        for package in packages:
            package_source = _materialized_package_source(mb, package)
            if package_source is None:
                raise FileNotFoundError(
                    f"privacy-filtered package source {package!r} is unavailable"
                )
            destination = scaffold_dst / package
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(
                package_source,
                destination,
                ignore=shutil.ignore_patterns(
                    ".git", "__pycache__", "*.pyc", "*.pyo"
                ),
                symlinks=False,
                ignore_dangling_symlinks=True,
            )
            _remove_private_paths(package, destination)
            created.append(package)

    for op in ops:
        kind = op.get("op")
        rel = op.get("file") or ""
        content = op.get("content", "")
        if not rel:
            continue
        try:
            _safe_rel_path(rel, field="mid_edit file")
        except ValueError as exc:
            _warn(f"{ctx.task_id}: skipping unsafe scaffold path {rel!r}: {exc}")
            continue

        if kind == "create":
            file_texts[rel] = _content_lines(content)
        elif kind == "replace":
            lines = _load_target(rel)
            if lines is None:
                _warn(
                    f"{ctx.task_id}: mid_edit replace target {rel!r} not found "
                    "in vendor source — skipping (workspace fidelity reduced)"
                )
                continue
            s = int(op["start_line"]) - 1
            e = _end_index(lines, int(op["end_line"]))
            # Split into per-line elements so subsequent ops' 1-indexed line
            # numbers stay consistent with native MLS-Bench
            # (src/mlsbench/agent/tools.py:3825-3836).
            new_lines = _content_lines(content)
            file_texts[rel] = lines[:s] + new_lines + lines[e:]
        elif kind == "insert":
            lines = _load_target(rel)
            if lines is None:
                _warn(
                    f"{ctx.task_id}: mid_edit insert target {rel!r} not found "
                    "in vendor source — skipping"
                )
                continue
            after = int(op.get("after_line", 0))
            new_lines = _content_lines(content)
            file_texts[rel] = lines[:after] + new_lines + lines[after:]
        elif kind == "delete":
            lines = _load_target(rel)
            if lines is None:
                _warn(
                    f"{ctx.task_id}: mid_edit delete target {rel!r} not found "
                    "in vendor source — skipping"
                )
                continue
            start, end = _delete_bounds(op)
            s = start - 1
            e = _end_index(lines, end)
            file_texts[rel] = lines[:s] + lines[e:]
        else:
            _warn(f"{ctx.task_id}: unknown mid_edit op kind {kind!r} on {rel!r}; skipping")

    if not clean_scaffold:
        for pkg in _task_packages(config):
            if _same_package_name(pkg, ctx.package):
                continue
            pkg_src = _materialized_package_source(mb, pkg)
            if pkg_src is None:
                _warn(
                    f"{ctx.task_id}: secondary package {pkg!r} not found in vendor source; "
                    "multi-package workspace fidelity reduced"
                )
                continue
            dst = scaffold_dst / pkg
            if dst.exists():
                shutil.rmtree(dst)
            try:
                shutil.copytree(
                    pkg_src,
                    dst,
                    ignore=shutil.ignore_patterns(
                        ".git", "__pycache__", "*.pyc", "*.pyo"
                    ),
                    symlinks=False,
                    ignore_dangling_symlinks=True,
                )
            except shutil.Error as exc:
                _warn(
                    f"{ctx.task_id}: secondary package {pkg!r} copytree had "
                    f"{len(exc.args[0])} non-fatal error(s); continuing"
                )
            created.append(pkg)

    # Materialize everything we touched into scaffold_dst.
    for rel, lines in file_texts.items():
        try:
            rel = _safe_rel_str(rel, field="mid_edit file")
            if _path_is_private(rel, private_paths):
                continue
            dst = _safe_join(scaffold_dst, rel, field="mid_edit file")
        except ValueError as exc:
            _warn(f"{ctx.task_id}: skipping unsafe scaffold path {rel!r}: {exc}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        text = "".join(lines)
        _assert_python_syntax(
            text,
            rel_path=rel,
            origin="mid_edit.py (and underlying pre_edit.py)",
            task_id=ctx.task_id,
        )
        dst.write_text(text)
        created.append(rel)

    if clean_scaffold:
        for package in packages:
            destination = scaffold_dst / package
            if destination.exists():
                _remove_private_paths(package, destination)
    return created


# Cache materialized package source dirs (vendor source + pre_edit applied)
# so we don't redo the copy for every task in the same adapter run.
_PKG_SRC_CACHE: dict[tuple[str, str], Path] = {}


def _materialized_package_source(mb: MlsBenchRoot, pkg: str) -> Path | None:
    """Return a directory containing the package source with package-level
    pre_edit.py applied. If no pre_edit.py exists, returns the vendor source
    path directly.
    """
    root_key = str(mb.root.resolve())
    cache_key = (root_key, pkg)
    if cache_key in _PKG_SRC_CACHE:
        return _PKG_SRC_CACHE[cache_key]
    pkg_src = mb.package_src(pkg)
    if pkg_src is None:
        return None
    pre_py = mb.pkg_configs_dir / pkg / "pre_edit.py"
    if not pre_py.exists() and mb.pkg_configs_dir.is_dir():
        norm = _normalize_pkg_name(pkg)
        for cand in mb.pkg_configs_dir.iterdir():
            if cand.is_dir() and _normalize_pkg_name(cand.name) == norm:
                pre_py = cand / "pre_edit.py"
                break
    if not pre_py.exists():
        _PKG_SRC_CACHE[cache_key] = pkg_src
        return pkg_src
    # Materialize once into an adapter-cache dir (per mb root so concurrent
    # runs against different checkouts don't collide).
    import hashlib
    import tempfile
    root_hash = hashlib.sha1(root_key.encode()).hexdigest()[:10]
    cache_root = Path(tempfile.gettempdir()) / "mls-bench-adapter-pre-edit" / root_hash
    cache_root.mkdir(parents=True, exist_ok=True)
    dst = cache_root / pkg
    if dst.exists():
        shutil.rmtree(dst)
    try:
        shutil.copytree(
            pkg_src, dst,
            ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
            symlinks=False,
            ignore_dangling_symlinks=True,
        )
    except shutil.Error as exc:
        # Some vendored packages contain broken symlinks (e.g. SMPyBandits/
        # docs/paper/plots/paper -> nonexistent). Log and continue; the
        # dangling links don't affect rendered tasks.
        _warn(
            f"{pkg}: shutil.copytree encountered "
            f"{len(exc.args[0])} non-fatal error(s); continuing"
        )
    pre_ops = _load_ops_file(pre_py)
    for op in pre_ops:
        rel = op.get("file") or ""
        if not rel:
            continue
        sub = _subpath_under_package(rel, pkg)
        target = dst / sub
        kind = op.get("op")
        content = op.get("content", "")
        if kind == "create":
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("".join(_content_lines(content)))
        elif kind == "replace" and target.exists():
            lines = target.read_text().splitlines(keepends=True)
            s = int(op["start_line"]) - 1
            e = _end_index(lines, int(op["end_line"]))
            target.write_text("".join(lines[:s] + _content_lines(content) + lines[e:]))
        elif kind == "insert":
            lines = target.read_text().splitlines(keepends=True) if target.exists() else []
            after = int(op.get("after_line", 0))
            target.write_text("".join(lines[:after] + _content_lines(content) + lines[after:]))
        elif kind == "delete" and target.exists():
            lines = target.read_text().splitlines(keepends=True)
            start, end = _delete_bounds(op)
            s = start - 1
            e = _end_index(lines, end)
            target.write_text("".join(lines[:s]) + "".join(lines[e:]))
    _PKG_SRC_CACHE[cache_key] = dst
    return dst


def _adapt_scripts_for_b200(scripts_dir: Path) -> None:
    """Adapt eval scripts for B200 (≈ 2× H100): halve TP, boost memory util.

    Applied in-place after copying scripts to the rendered task directory.
    Only modifies shell scripts that contain known multi-GPU patterns.
    """
    if not scripts_dir.exists():
        return
    import re as _re
    for sh in scripts_dir.glob("*.sh"):
        text = sh.read_text()
        orig = text
        # tensor_model_parallel_size=2 → 1 (verl/vLLM RL tasks)
        text = _re.sub(
            r"tensor_model_parallel_size=2\b",
            "tensor_model_parallel_size=1",
            text,
        )
        # gpu_memory_utilization=0.4 → 0.5 (more VRAM on B200)
        text = _re.sub(
            r"gpu_memory_utilization=0\.4\b",
            "gpu_memory_utilization=0.5",
            text,
        )
        # Hardcoded nproc_per_node=4 → 2 (nanogpt pretrain 4-GPU scripts)
        text = _re.sub(
            r"nproc_per_node=4\b",
            "nproc_per_node=2",
            text,
        )
        # Hardcoded nproc_per_node=8 → 2
        text = _re.sub(
            r"nproc_per_node=8\b",
            "nproc_per_node=2",
            text,
        )
        if text != orig:
            sh.write_text(text)


def _stage_verifier_assets(
    mb: MlsBenchRoot,
    ctx: TaskContext,
    tests_dir: Path,
    task_dir: Path,
    *,
    scaffold_dir: Path,
    config: dict | None = None,
    skip_pristine: bool = False,
    mangrove: bool = False,
    gpu_backend: str = "h20",
) -> None:
    """Stage the per-task verifier-only files under tests/.

    Harbor mounts this directory at /tests/ only during verification, so
    nothing here leaks to the agent during its work session.

    Layout produced:
      tests/meta/config.json
      tests/meta/parser.py
      tests/meta/score_spec.py
      tests/meta/leaderboard.csv
      tests/meta/budget_check.py        (only if the task has one)
      tests/meta/edits/*                (needed by budget_check.py)
      tests/meta/scripts/*              (needed by some budget_check.py files)
      tests/meta/task_id                (single-line)
      tests/meta/package                (single-line)
      tests/meta/workdir                (single-line)
      tests/meta/package_envs.json
      tests/eval/scripts/*.sh           (all eval scripts, visible + hidden)
    """
    meta = tests_dir / "meta"
    meta.mkdir(exist_ok=True)
    for fname in ("parser.py", "score_spec.py", "leaderboard.csv"):
        src = task_dir / fname
        if src.exists():
            shutil.copy2(src, meta / fname)
    cfg_to_write = deepcopy(config if config is not None else ctx.config)
    for test_cmd in cfg_to_write.get("test_cmds", []):
        test_cmd.pop("hidden", None)
    mangrove_mode = mangrove or skip_pristine
    if mangrove_mode and gpu_backend != "h20":
        for tc in cfg_to_write.get("test_cmds", []):
            raw = tc.get("time")
            if raw:
                secs = _parse_time(raw) * 3
                tc["time"] = f"{secs // 3600}:{(secs % 3600) // 60:02d}:{secs % 60:02d}"
    (meta / "config.json").write_text(
        json.dumps(cfg_to_write, indent=2) + "\n"
    )
    if (task_dir / "budget_check.py").exists():
        shutil.copy2(task_dir / "budget_check.py", meta / "budget_check.py")
    _stage_verifier_data_deps(mb, task_dir, meta, cfg_to_write)
    _stage_verifier_only_package_files(mb, ctx, meta, cfg_to_write)
    (meta / "task_id").write_text(ctx.task_id + "\n")
    (meta / "package").write_text(ctx.package + "\n")
    (meta / "workdir").write_text(ctx.pkg_config.get("workdir", "/workspace") + "\n")
    cfg = config if config is not None else ctx.config
    if mangrove_mode:
        res_for_gpu = _mangrove_resources(ctx.pkg_config, cfg, gpu_backend=gpu_backend)
    else:
        res_for_gpu = _resources(ctx.pkg_config, cfg)
    (meta / "gpu_count").write_text(str(res_for_gpu["gpus"]) + "\n")
    gpu_cap = res_for_gpu.get("gpu_compute_cap", 1)
    if gpu_cap > 1:
        (meta / "gpu_compute_cap").write_text(str(gpu_cap) + "\n")
    package_envs: dict[str, dict] = {}
    for tc in ctx.config.get("test_cmds", []):
        pkg = tc.get("package")
        if not pkg or pkg in package_envs:
            continue
        package_envs[pkg] = _load_pkg_config(mb, pkg).get("env") or {}
    if ctx.package and ctx.package not in package_envs:
        package_envs[ctx.package] = ctx.pkg_config.get("env") or {}
    (meta / "package_envs.json").write_text(json.dumps(package_envs, indent=2))

    edits_src = task_dir / "edits"
    if edits_src.exists():
        shutil.copytree(
            edits_src,
            meta / "edits",
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )

    scripts_src = task_dir / "scripts"
    if scripts_src.exists():
        shutil.copytree(
            scripts_src,
            meta / "scripts",
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )

    # Eval scripts.
    if scripts_src.exists():
        shutil.copytree(
            scripts_src,
            tests_dir / "eval" / "scripts",
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )

    # Mangrove B200 adaptation: halve GPU counts and double batch sizes in
    # eval scripts.  B200 ≈ 2× H100, so we run with half the physical GPUs.
    # Skip for H20 — it's ≈ H100 so original scripts work as-is.
    if mangrove_mode and gpu_backend != "h20":  # Mangrove B200 mode
        _adapt_scripts_for_b200(tests_dir / "eval" / "scripts")
        _adapt_scripts_for_b200(meta / "scripts")

    # mlsbench source tree — required by parser.py / score_spec.py / score_task.py
    # to import mlsbench.scoring.* and mlsbench.agent.parsers. NOT baked into
    # the base image because agent shell would see it; instead we ship it here
    # where Harbor mounts it only at verify time. score_task.py adds
    # /tests/mlsbench_src to sys.path.
    mlsbench_src = mb.src_dir / "mlsbench"
    if mlsbench_src.exists():
        shutil.copytree(
            mlsbench_src, tests_dir / "mlsbench_src" / "mlsbench",
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )

    # Pristine diff baseline for the edit-range guard. NOT in the image:
    # agent runs as root and could overwrite an image-baked baseline. Lives
    # under tests/meta/pristine/ + tests/meta/pristine_manifest.json so
    # Harbor uploads it only at verify time. See score_task.py::cmd_guard.
    if not skip_pristine:
        _stage_pristine_assets(
            mb, ctx, meta, scaffold_dir=scaffold_dir,
            config=config if config is not None else ctx.config,
        )


def _stage_pristine_assets(
    mb: MlsBenchRoot,
    ctx: TaskContext,
    meta: Path,
    *,
    scaffold_dir: Path,
    config: dict,
) -> None:
    """Stage the verifier-side pristine diff baseline.

    Produces two artifacts under ``meta/`` (= ``tests/meta/``):
      - ``pristine/<rel>`` — full bytes for every file declared in
        ``config.json::files[]``. Needed verbatim by
        ``score_task.py::_check_editable_only`` for content-based fixed
        segment matching.
      - ``pristine_manifest.json`` — ``{rel_path: sha256}`` for every file
        under any guarded package prefix. Lets the guard catch any
        modification to non-declared files (and deletions / creations)
        without paying the cost of shipping the whole package source.

    Both come from a synthetic "agent-start workspace" = pre_edit-applied
    package source overlaid by the per-task mid_edit scaffold. The same
    state that ``/workspace/<pkg>/`` is initialized to inside the container.
    """
    pristine_root = meta / "pristine"
    manifest: dict[str, str] = {}

    # Build agent-start state per package. Scaffold takes precedence over
    # base source on a per-file basis (mid_edit overlay).
    packages = list(_task_packages(config))
    if ctx.package and ctx.package not in packages:
        packages.append(ctx.package)

    # Index scaffold contents up front: rel-path (workspace-relative,
    # POSIX) -> absolute path on the host. The scaffold dir mirrors the
    # workspace layout (its first path component is the package dir).
    scaffold_index: dict[str, Path] = {}
    if scaffold_dir.is_dir():
        import stat as _stat
        for fp in scaffold_dir.rglob("*"):
            try:
                st = fp.lstat()
            except OSError:
                continue
            if not _stat.S_ISREG(st.st_mode):
                continue
            rel = fp.relative_to(scaffold_dir).as_posix()
            scaffold_index[rel] = fp

    manifest_mode = str(config.get("pristine_manifest_mode", "package")).lower()
    if manifest_mode not in {"package", "scaffold"}:
        raise ValueError(
            f"unsupported pristine_manifest_mode={manifest_mode!r} for "
            f"{ctx.task_id!r}; expected 'package' or 'scaffold'"
        )
    private_paths = (
        _verifier_only_package_paths(config)
        | _agent_pruned_package_paths(config)
    )

    declared_files = {
        _safe_rel_str(str(f["filename"]), field="files[].filename")
        for f in config.get("files", [])
        if f.get("filename")
    }

    seen_rel: set[str] = set()

    def _hash_bytes(b: bytes) -> str:
        return hashlib.sha256(b).hexdigest()

    def _emit(rel: str, content_bytes: bytes) -> None:
        if rel in seen_rel or _path_is_private(rel, private_paths):
            return
        seen_rel.add(rel)
        manifest[rel] = _hash_bytes(content_bytes)
        if rel in declared_files:
            try:
                dst = _safe_join(pristine_root, rel, field="pristine path")
            except ValueError:
                return
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(content_bytes)

    if manifest_mode == "package":
        for pkg in packages:
            pkg_src = _materialized_package_source(mb, pkg)
            if pkg_src is None:
                raise FileNotFoundError(
                    f"_stage_pristine_assets: package source for {pkg!r} not found; "
                    f"cannot guard task {ctx.task_id!r}"
                )
            pkg_src_resolved = pkg_src.resolve()
            for fp in pkg_src.rglob("*"):
                try:
                    st = fp.lstat()
                except OSError:
                    continue
                if not stat.S_ISREG(st.st_mode):
                    continue
                try:
                    fp.resolve().relative_to(pkg_src_resolved)
                except ValueError:
                    continue
                subpath = fp.relative_to(pkg_src)
                if any(part in {".git", "__pycache__"} for part in subpath.parts):
                    continue
                if fp.suffix in {".pyc", ".pyo"}:
                    continue
                rel = f"{pkg}/{subpath.as_posix()}"
                scaffold_fp = scaffold_index.get(rel)
                try:
                    _emit(
                        rel,
                        scaffold_fp.read_bytes() if scaffold_fp is not None else fp.read_bytes(),
                    )
                except OSError:
                    continue

    # Scaffold-only files: created by mid_edit `create` ops or copied as
    # part of a secondary package not iterated above.
    for rel, fp in scaffold_index.items():
        if rel in seen_rel:
            continue
        try:
            _emit(rel, fp.read_bytes())
        except OSError:
            continue

    missing_declared = declared_files - seen_rel
    if missing_declared:
        raise FileNotFoundError(
            f"{ctx.task_id}: pristine baseline is missing declared file(s): "
            f"{sorted(missing_declared)}"
        )

    if not manifest:
        return  # nothing to guard (e.g. tasks with no guarded files)
    (meta / "pristine_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def _allowed_to_protected(allowed_ranges: list[list[int]]) -> list[list[int]]:
    if not allowed_ranges:
        return [[-1, -1]]
    if any(r[0] == -1 and r[1] == -1 for r in allowed_ranges):
        return []

    sorted_r = sorted(allowed_ranges, key=lambda x: x[0])
    protected: list[list[int]] = []
    if sorted_r[0][0] > 1:
        protected.append([1, sorted_r[0][0] - 1])
    for i in range(len(sorted_r) - 1):
        prev_end = sorted_r[i][1]
        if prev_end == -1:
            break
        gap_start = prev_end + 1
        gap_end = sorted_r[i + 1][0] - 1
        if gap_start <= gap_end:
            protected.append([gap_start, gap_end])
    last_end = sorted_r[-1][1]
    if last_end != -1:
        protected.append([last_end + 1, -1])
    return protected


def _protected_to_allowed(protected_ranges: list[list[int]]) -> list[dict]:
    if not protected_ranges:
        return [{"start": -1, "end": -1}]
    if any(r[0] == -1 and r[1] == -1 for r in protected_ranges):
        return []

    protected = sorted(protected_ranges, key=lambda x: x[0])
    allowed: list[dict] = []
    if protected[0][0] > 1:
        allowed.append({"start": 1, "end": protected[0][0] - 1})
    for i in range(len(protected) - 1):
        prev_end = protected[i][1]
        if prev_end == -1:
            break
        gap_start = prev_end + 1
        gap_end = protected[i + 1][0] - 1
        if gap_start <= gap_end:
            allowed.append({"start": gap_start, "end": gap_end})
    last_end = protected[-1][1]
    if last_end != -1:
        allowed.append({"start": last_end + 1, "end": -1})
    return allowed


def _shift_protected_ranges(
    protected_ranges: list[list[int]],
    change_after_line: int,
    delta: int,
) -> None:
    """Mirror WorkspaceTools._shift_ranges_for_pre_edit."""
    if delta == 0:
        return
    for r in protected_ranges:
        if r[0] == -1 and r[1] == -1:
            continue
        if r[0] > change_after_line:
            r[0] = max(1, r[0] + delta)
        if r[1] != -1 and r[1] > change_after_line:
            r[1] = max(1, r[1] + delta)


def _ops_for_task_workspace(mb: MlsBenchRoot, ctx: TaskContext) -> list[dict]:
    ops: list[dict] = []
    for pkg in _task_packages(ctx.config):
        pre_py = mb.pkg_configs_dir / pkg / "pre_edit.py"
        if not pre_py.exists() and mb.pkg_configs_dir.is_dir():
            norm = _normalize_pkg_name(pkg)
            for cand in mb.pkg_configs_dir.iterdir():
                if cand.is_dir() and _normalize_pkg_name(cand.name) == norm:
                    pre_py = cand / "pre_edit.py"
                    break
        if pre_py.exists():
            ops.extend(_load_ops_file(pre_py))

    mid_py = mb.tasks_dir / ctx.task_id / "edits" / "mid_edit.py"
    if mid_py.exists():
        ops.extend(_load_ops_file(mid_py))
    return ops


def _config_with_shifted_edit_ranges(mb: MlsBenchRoot, ctx: TaskContext) -> dict:
    """Return config.json with edit ranges in the post-setup workspace line space.

    Native WorkspaceTools builds protected ranges from config.json, then shifts
    those protected boundaries after every pre_edit and mid_edit mutation
    (src/mlsbench/agent/tools.py:752, 3828, 3837, 3845, 3849).
    """
    import copy

    config = copy.deepcopy(ctx.config)
    protected_by_file: dict[str, list[list[int]]] = {}
    for entry in config.get("files", []):
        if "edit" not in entry:
            continue
        filename = _safe_rel_str(str(entry["filename"]), field="files[].filename")
        allowed = [[int(r["start"]), int(r["end"])] for r in (entry.get("edit") or [])]
        protected_by_file[filename] = _allowed_to_protected(allowed)

    if not protected_by_file:
        return config

    for op in _ops_for_task_workspace(mb, ctx):
        rel = op.get("file") or ""
        if rel not in protected_by_file:
            continue
        kind = op.get("op")
        if kind == "replace":
            start = int(op["start_line"])
            end = int(op["end_line"])
            old_count = 0 if end == -1 else end - start + 1
            change_after_line = end if end != -1 else start - 1
            delta = _line_count(str(op.get("content", ""))) - old_count
        elif kind == "insert":
            change_after_line = int(op.get("after_line", 0))
            delta = _line_count(str(op.get("content", "")))
        elif kind == "delete":
            start, end = _delete_bounds(op)
            old_count = 1 if end == -1 else end - start + 1
            change_after_line = start - 1
            delta = -old_count
        else:
            continue
        _shift_protected_ranges(protected_by_file[rel], change_after_line, delta)

    for entry in config.get("files", []):
        filename = _safe_rel_str(str(entry["filename"]), field="files[].filename")
        if filename in protected_by_file:
            entry["edit"] = _protected_to_allowed(protected_by_file[filename])
    return config




def _apply_ops_to_text(template_text: str, ops: list[dict], target_file: str) -> str:
    """Apply just the ops that target `target_file` to `template_text`.

    Mirrors `BaseAgent._apply_edit_ops` (`src/mlsbench/agent/base.py:296`).
    Supports replace/insert/delete; ignores ops for other files.
    """
    lines = template_text.splitlines(keepends=True)
    relevant = [op for op in ops if _op_file_matches(op, target_file)]
    for op in relevant:
        kind = op.get("op")
        content = op.get("content", "")
        content_lines = _content_lines(content)
        if kind == "replace":
            s = int(op["start_line"]) - 1
            e = _end_index(lines, int(op["end_line"]))
            lines[s:e] = content_lines
        elif kind == "insert":
            after = int(op.get("after_line", 0))
            lines[after:after] = content_lines
        elif kind == "delete":
            start, end = _delete_bounds(op)
            s = start - 1
            e = _end_index(lines, end)
            del lines[s:e]
    return "".join(lines)


def _adjusted_edit_ranges(
    edit_ranges: list[dict],
    ops: list[dict],
    target_file: str,
    original_line_count: int,
) -> list[dict]:
    """Mirror `BaseAgent._adjusted_edit_ranges` for already-loaded ops."""
    relevant = [op for op in ops if _op_file_matches(op, target_file)]
    if not relevant:
        return list(edit_ranges)

    bounds: list[list[int] | None] = []
    for r in edit_ranges:
        if int(r["start"]) == -1:
            bounds.append(None)
        else:
            bounds.append([int(r["start"]), int(r["end"])])

    for op in relevant:
        kind = op.get("op")
        if kind == "replace":
            op_s = int(op["start_line"])
            op_e = int(op["end_line"])
            if op_e == -1:
                op_e = original_line_count
            delta = _line_count(str(op.get("content", ""))) - (op_e - op_s + 1)
        elif kind == "insert":
            op_s = int(op["after_line"])
            op_e = op_s
            delta = _line_count(str(op.get("content", "")))
        elif kind == "delete":
            op_s, op_e = _delete_bounds(op)
            if op_e == -1:
                op_e = original_line_count
            delta = -(op_e - op_s + 1)
        else:
            continue

        for b in bounds:
            if b is None:
                continue
            if kind == "replace":
                if op_s >= b[0] and op_e <= b[1]:
                    b[1] += delta
                elif op_e < b[0]:
                    b[0] += delta
                    b[1] += delta
            elif kind == "insert":
                if op_s >= b[0] and op_s <= b[1]:
                    b[1] += delta
                elif op_s < b[0]:
                    b[0] += delta
                    b[1] += delta
            elif kind == "delete":
                if op_s >= b[0] and op_e <= b[1]:
                    b[1] += delta
                elif op_e < b[0]:
                    b[0] += delta
                    b[1] += delta

    return [
        {"start": -1, "end": -1} if b is None else {"start": b[0], "end": b[1]}
        for b in bounds
    ]


def _starting_workspace_text(
    mb: MlsBenchRoot,
    ctx: TaskContext,
    rel_filename: str,
) -> str | None:
    """Return the initial contents of an editable file in the agent's workspace.

    This mirrors BaseAgent.setup_workspace(): vendor source with package-level
    pre_edit applied, followed by the task's mid_edit ops.
    """
    rel_filename = _safe_rel_str(rel_filename, field="files[].filename")
    pkg = _package_from_rel(ctx.config, ctx.package, rel_filename)
    pkg_src = _materialized_package_source(mb, pkg)
    text: str | None = None
    if pkg_src is not None:
        sub = _subpath_under_package(rel_filename, pkg)
        template_path = (pkg_src / sub).resolve()
        try:
            template_path.relative_to(pkg_src.resolve())
        except ValueError:
            return None
        if template_path.exists():
            text = template_path.read_text()

    task_dir = mb.tasks_dir / ctx.task_id
    mid_edit = task_dir / "edits" / "mid_edit.py"
    if mid_edit.exists():
        ops = _load_ops_file(mid_edit)
        lines = text.splitlines(keepends=True) if text is not None else []
        exists = text is not None
        touched = False
        for op in ops:
            if not _op_file_matches(op, rel_filename):
                continue
            touched = True
            kind = op.get("op")
            content = str(op.get("content", ""))
            if kind == "create":
                lines = _content_lines(content)
                exists = True
            elif kind == "replace":
                if not exists:
                    raise FileNotFoundError(f"mid_edit target not found: {rel_filename}")
                s = int(op["start_line"]) - 1
                e = _end_index(lines, int(op["end_line"]))
                lines[s:e] = _content_lines(content)
            elif kind == "insert":
                if not exists:
                    raise FileNotFoundError(f"mid_edit target not found: {rel_filename}")
                after = int(op.get("after_line", 0))
                lines[after:after] = _content_lines(content)
            elif kind == "delete":
                if not exists:
                    raise FileNotFoundError(f"mid_edit target not found: {rel_filename}")
                start, end = _delete_bounds(op)
                s = start - 1
                e = _end_index(lines, end)
                del lines[s:e]
        if touched:
            return "".join(lines) if exists else None
    return text


def _baseline_sections(
    mb: MlsBenchRoot,
    ctx: TaskContext,
    config: dict | None = None,
    context_lines: int = 3,
) -> tuple[list[dict], list[str]]:
    """Render all baselines' editable regions, matching `base.py:425-498`.

    Each item: {name, filename, code} with `code` being numbered, sliced to the
    editable ranges with `context_lines` of surrounding context.
    """
    config = config or ctx.config
    baselines = config.get("baselines") or {}
    if not bool(config.get("rigorous_codebase", False)) or not baselines:
        return [], []
    task_dir = mb.tasks_dir / ctx.task_id

    sections: list[dict] = []
    degraded: list[str] = []

    def warn(reason: str) -> None:
        msg = f"{ctx.task_id}: baseline prompt degraded: {reason}"
        degraded.append(reason)
        _warn(msg)

    for entry in config.get("files", []):
        ranges = entry.get("edit") or []
        if not ranges:
            continue
        try:
            bl_filename = _safe_rel_str(str(entry["filename"]), field="files[].filename")
        except ValueError as exc:
            warn(f"unsafe editable filename {entry.get('filename')!r}: {exc}")
            continue
        try:
            template_text = _starting_workspace_text(mb, ctx, bl_filename)
        except Exception as exc:
            template_text = None
            warn(f"`{bl_filename}` could not be read: {exc}")
        if template_text is None:
            warn(f"`{bl_filename}` source file was not found")
            continue

        for bl_name, bl_cfg in baselines.items():
            edit_ops_rel = bl_cfg.get("edit_ops")
            if not edit_ops_rel:
                warn(f"`{bl_name}` has no edit_ops entry")
                continue
            try:
                ops_py = _safe_join(task_dir, edit_ops_rel, field="baseline edit_ops")
            except ValueError as exc:
                warn(f"`{bl_name}` edit_ops path is unsafe ({edit_ops_rel!r}): {exc}")
                continue
            if not ops_py.exists():
                warn(f"`{bl_name}` edit_ops file is missing: {edit_ops_rel}")
                continue
            try:
                ops = _load_ops_file(ops_py)
            except Exception as exc:
                warn(f"`{bl_name}` edit_ops could not be loaded ({edit_ops_rel}): {exc}")
                continue
            if not any(_op_file_matches(op, bl_filename) for op in ops):
                continue
            try:
                rendered = _apply_ops_to_text(template_text, ops, bl_filename)
            except Exception:
                warn(f"`{bl_name}` could not be applied to `{bl_filename}`")
                continue
            if rendered.rstrip() == template_text.rstrip():
                continue
            all_lines = rendered.splitlines()
            adj_ranges = _adjusted_edit_ranges(
                ranges,
                ops,
                bl_filename,
                original_line_count=len(template_text.splitlines()),
            )

            # Slice to editable ranges with surrounding context.
            parts: list[str] = []
            for r in adj_ranges:
                s, e = int(r["start"]), int(r["end"])
                if s == -1 and e == -1:
                    numbered = "\n".join(f"{i + 1:6d}: {ln}" for i, ln in enumerate(all_lines))
                    parts.append(numbered)
                    continue
                cs = max(0, s - 1 - context_lines)
                ce = min(len(all_lines), e + context_lines)
                slice_lines = all_lines[cs:ce]
                numbered = "\n".join(
                    f"{cs + i + 1:6d}: {ln}" for i, ln in enumerate(slice_lines)
                )
                parts.append(f"Lines {s}–{e}:\n{numbered}")

            if parts:
                sections.append({
                    "name": bl_name,
                    "filename": bl_filename,
                    "code": "\n\n".join(parts),
                })
    return sections, degraded


def _language_for(filename: str) -> str:
    if filename.endswith(".sh"):
        return "bash"
    if filename.endswith((".json", ".jsonl")):
        return "json"
    if filename.endswith((".toml", ".tml")):
        return "toml"
    if filename.endswith((".yaml", ".yml")):
        return "yaml"
    if filename.endswith(".md"):
        return "markdown"
    return "python"


def _format_numbered_ranges(
    filename: str,
    text: str,
    ranges: list[dict],
    *,
    max_lines: int = 500,
    max_chars: int = 60_000,
) -> tuple[str, bool]:
    lines = text.splitlines()
    file_sections: list[str] = []
    truncated = False
    remaining_lines = max_lines
    remaining_chars = max_chars

    for rng in ranges:
        if remaining_lines <= 0 or remaining_chars <= 0:
            truncated = True
            break
        start = int(rng["start"])
        end = int(rng["end"])
        if start == -1 and end == -1:
            selected = list(enumerate(lines, start=1))
            header = ""
        else:
            end_idx = len(lines) if end == -1 else end
            selected = list(enumerate(lines[start - 1 : end_idx], start=start))
            header = f"Lines {start}-{end}:\n"

        rendered_lines: list[str] = []
        for line_no, line in selected:
            candidate = f"{line_no:6d}: {line}"
            if len(rendered_lines) >= remaining_lines or len(candidate) + 1 > remaining_chars:
                truncated = True
                break
            rendered_lines.append(candidate)
            remaining_chars -= len(candidate) + 1
        remaining_lines -= len(rendered_lines)
        if rendered_lines:
            file_sections.append(header + "\n".join(rendered_lines))

    if truncated:
        file_sections.append(
            f"[truncated: showing at most {max_lines} lines / {max_chars} bytes from {filename}]"
        )
    return "\n\n".join(file_sections), truncated


def _read_sections(
    mb: MlsBenchRoot,
    ctx: TaskContext,
    config: dict | None = None,
) -> tuple[list[dict], list[str]]:
    config = config or ctx.config
    sections: list[dict] = []
    degraded: list[str] = []
    rigorous = bool(config.get("rigorous_codebase", False))
    baselines = config.get("baselines") or {}

    for entry in config.get("files", []):
        read_ranges = entry.get("read") or []
        if not read_ranges:
            continue
        try:
            filename = _safe_rel_str(str(entry["filename"]), field="files[].filename")
        except ValueError as exc:
            reason = f"read context skipped unsafe filename {entry.get('filename')!r}: {exc}"
            degraded.append(reason)
            _warn(f"{ctx.task_id}: {reason}")
            continue

        edit_ranges = entry.get("edit") or []
        if rigorous and baselines and not edit_ranges:
            continue
        if not edit_ranges:
            edit_note = "READ-ONLY — do not edit"
        else:
            range_strs = [
                "entire file" if int(r["start"]) == -1 else f"lines {int(r['start'])}–{int(r['end'])}"
                for r in edit_ranges
            ]
            edit_note = f"EDITABLE — {', '.join(range_strs)} only"

        try:
            text = _starting_workspace_text(mb, ctx, filename)
        except Exception as exc:
            text = None
            reason = f"`{filename}` read context could not be loaded: {exc}"
            degraded.append(reason)
            _warn(f"{ctx.task_id}: {reason}")
        if text is None:
            reason = f"`{filename}` read context source file was not found"
            degraded.append(reason)
            _warn(f"{ctx.task_id}: {reason}")
            continue

        code, _ = _format_numbered_ranges(filename, text, read_ranges)
        if code:
            sections.append({
                "filename": filename,
                "edit_note": edit_note,
                "language": _language_for(filename),
                "code": code,
            })
    return sections, degraded


def _domain_tags(task_id: str) -> list[str]:
    """Domain prefix → Harbor tags."""
    prefix = task_id.split("-", 1)[0]
    return {
        "ai4bio": ["ml-research", "biology"],
        "ai4sci": ["ml-research", "science"],
        "causal": ["ml-research", "causal-inference"],
        "cv": ["ml-research", "computer-vision"],
        "rl": ["ml-research", "reinforcement-learning"],
        "llm": ["ml-research", "language-model"],
        "agent": ["ml-research", "agent"],
        "nlp": ["ml-research", "nlp"],
    }.get(prefix, ["ml-research"])
