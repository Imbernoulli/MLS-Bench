"""Core adapter logic: read MLS-Bench task tree, emit Harbor task dirs."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import os
import re
import shutil
import stat
import statistics
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
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
    oracle_cmd_overrides: list[dict] = field(default_factory=list)


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
    ):
        self.output_dir = output_dir
        self.limit = limit
        self.overwrite = overwrite
        self.task_ids = task_ids
        self.mls_bench_root = mls_bench_root
        self.continue_on_error = continue_on_error

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
                out = render_task(mb, ctx, self.output_dir, overwrite=self.overwrite)
                print(f"[ok] {task_id} -> {out}")
                ok += 1
                task_dirs.append(out)
            except Exception as exc:
                failed.append((task_id, exc))
                # Drop any half-rendered staging dir (`<task>.inprogress`) so a
                # later manifest scan / tarball doesn't pick it up.
                stage = self.output_dir / (_harbor_safe_name(task_id) + ".inprogress")
                if stage.exists():
                    shutil.rmtree(stage, ignore_errors=True)
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
        # `compile` is stricter than `ast.parse`: it additionally rejects
        # module-level `return`/`break` etc that ast.parse silently accepts,
        # catching a wider class of malformed paste-ins.
        compile(text, rel_path, "exec")
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

    best_name: str | None = None
    best_score: float = float("-inf")
    any_scored = False
    for name in baselines:
        candidates = {name, f"baseline:{name}"}
        rows = [r for r in leaderboard if r.get("model") in candidates]
        if not rows:
            continue
        # Four-tier precedence matching native `evaluate.evaluate_task`
        # (src/mlsbench/scoring/evaluate.py:678-719):
        # final_mean > final_any > nonfinal_mean > any row. Within each tier
        # prefer the most metric-complete row, then latest by timestamp.
        final_mean = [r for r in rows
                      if r.get("seed") == "mean"
                      and str(r.get("is_final", "")).lower() == "true"]
        final_any = [r for r in rows
                     if str(r.get("is_final", "")).lower() == "true"]
        nonfinal_mean = [r for r in rows if r.get("seed") == "mean"]
        chosen = (
            _pick_row(final_mean)
            or _pick_row(final_any)
            or _pick_row(nonfinal_mean)
            or _pick_row(rows)
        )
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
    baseline_name = ""
    edit_ops: list[dict] = []
    oracle_cmd_overrides: list[dict] = []
    baselines = config.get("baselines") or {}
    if baselines:
        baseline_name = _pick_strongest_baseline(task_dir, config, leaderboard)
        baseline_cfg = baselines[baseline_name]
        rel = baseline_cfg.get("edit_ops")
        if rel:
            try:
                ops_py = _safe_join(task_dir, rel, field="baseline edit_ops")
                edit_ops = _load_ops_file(ops_py)
            except Exception as exc:
                raise RuntimeError(
                    f"{task_id}: could not load oracle baseline ops {rel}: {exc}"
                ) from exc
        # Native MLSBench applies edit_ops AND cmd independently (cli.py: the
        # `if baseline_config.get("cmd")` and `if baseline_config.get("edit_ops")`
        # blocks are separate). A baseline with BOTH — e.g. Time-Series baselines
        # whose edit_ops point Custom.py at the baseline model and whose cmd runs
        # that model with the right per-dataset hyperparameters — needs both. A
        # prior `elif` here dropped the cmd override whenever edit_ops existed, so
        # the oracle ran the original eval script with the agent's large default
        # hyperparameters and budget_check then rejected the run (all-zero TS
        # oracle). Emit the cmd override regardless of edit_ops.
        if baseline_cfg.get("cmd"):
            labels = baseline_cfg.get("labels")
            if labels is None:
                oracle_cmd_overrides.append({"label": "", "cmd": baseline_cfg["cmd"]})
            else:
                for label in labels:
                    oracle_cmd_overrides.append({"label": str(label), "cmd": baseline_cfg["cmd"]})

    return TaskContext(
        task_id=task_id,
        task_description=task_description,
        config=config,
        package=package,
        pkg_config=pkg_config,
        leaderboard_rows=leaderboard,
        chosen_baseline=baseline_name,
        baseline_edit_ops=edit_ops,
        oracle_cmd_overrides=oracle_cmd_overrides,
    )


def _oracle_cmd_overrides_token(ctx: TaskContext) -> str:
    payload = {
        "task_id": ctx.task_id,
        "baseline": ctx.chosen_baseline,
        "overrides": ctx.oracle_cmd_overrides,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


# --------------------------------------------------------------------------- #
# Render
# --------------------------------------------------------------------------- #

PREBUILT_DOCKER = "bohanlyu2022/mlsbench-{pkg}:latest"
# The harbor-flavored image inherits from the prebuilt one and bakes in the
# pristine package source, mlsbench src, and any data deps. See
# harbor_adapter/scripts/build_base_image.py.
HARBOR_BASE_DOCKER = "bohanlyu2022/mlsbench-harbor-{pkg}:latest"


def _harbor_safe_name(task_id: str) -> str:
    safe = re.sub(r"[^a-z0-9_-]", "-", task_id.lower())
    return f"mls-bench__{safe}"


def _agent_timeout_sec(config: dict) -> int:
    # The agent only edits; eval runs in the verifier. Use a flat cap proportional
    # to the visible eval time so harder tasks get longer to think, bounded
    # to keep cloud bills sane.
    visible_total = sum(
        _parse_time(tc.get("time", "0:30:00"))
        for tc in config.get("test_cmds", []) if not tc.get("hidden")
    )
    # 30 min minimum, 0.5× of eval time as the heuristic, 4h hard cap.
    return max(30 * 60, min(4 * 3600, visible_total // 2 or 30 * 60))


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


def _verifier_timeout_sec(config: dict) -> int:
    # Groups run sequentially, while all (test_cmd, seed) jobs inside a group
    # run in parallel subject to bin-packing. Charge wall time by each group's
    # slowest member; budget-check headroom still scales per test_cmd and seed.
    test_cmds = list(config.get("test_cmds", []) or [])
    grouped = _group_test_cmds(test_cmds)
    total = sum(
        max(_parse_time(tc.get("time", "0:30:00")) for tc in entries)
        for entries in grouped.values()
    )
    n_seeds = _seed_count(config)
    return total + 30 * 60 + 120 * len(test_cmds) * n_seeds


def _bin_pack_fractional_gpus(fractionals: list[float]) -> int:
    """Min 1.0-capacity bins needed to hold all fractional GPU jobs.

    First-fit-decreasing: place each (sorted-descending) job into the first bin
    that has room, otherwise open a new bin. This is the correct count of GPUs
    needed concurrently, vs the wrong `ceil(sum)` approximation (e.g. 9 jobs
    of 0.4 GPUs each → ceil(3.6)=4 but actually requires 5 bins because each
    bin holds at most floor(1/0.4)=2 such jobs).
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
            whole = 0
            fractionals: list[float] = []
            for tc in entries:
                compute = _test_cmd_compute(tc)
                if compute >= 1.0:
                    # `seeds` parallel copies of a whole-GPU job each take its
                    # own dedicated GPU(s).
                    whole += n_seeds * max(1, math.ceil(compute))
                elif compute > 0.0:
                    # Each (entry, seed) is a separate concurrent job competing
                    # for fractional GPU space — `n_seeds` copies of the same
                    # entry. Bin-packed below.
                    fractionals.extend([compute] * n_seeds)
            peak_gpus = max(peak_gpus, whole + _bin_pack_fractional_gpus(fractionals))
        gpus = max(1, peak_gpus)
    return dict(cpus=cpus, memory_mb=memory_mb, storage_mb=storage_mb, gpus=gpus)


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
) -> Path:
    final_dir = out_root / _harbor_safe_name(ctx.task_id)
    if final_dir.exists() and not overwrite:
        return final_dir

    # Render into a sibling staging dir, rename on success. Partial renders
    # (e.g. when _stage_pristine_assets fails for missing vendor packages)
    # never leave a malformed task directory that downstream tooling
    # (dataset.toml writer, Harbor) would pick up.
    out_dir = out_root / (final_dir.name + ".inprogress")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    if final_dir.exists():
        shutil.rmtree(final_dir)
    out_dir.mkdir(parents=True)

    task_dir = mb.tasks_dir / ctx.task_id

    pkg_workdir = ctx.pkg_config.get("workdir", "/workspace")
    res = _resources(ctx.pkg_config, ctx.config)
    effective_config = _config_with_shifted_edit_ranges(mb, ctx)

    visible_test_cmds = list(effective_config.get("test_cmds", []))
    baseline_sections, baseline_warnings = _baseline_sections(mb, ctx, config=effective_config)
    read_sections, read_warnings = _read_sections(mb, ctx, config=effective_config)
    oracle_cmd_overrides_token = _oracle_cmd_overrides_token(ctx)

    template_ctx = {
        "task_id": ctx.task_id,
        "task_description": ctx.task_description,
        "package": ctx.package,
        "workdir": pkg_workdir,
        "base_image": HARBOR_BASE_DOCKER.format(pkg=ctx.package.lower()),
        "agent_timeout_sec": _agent_timeout_sec(ctx.config),
        "verifier_timeout_sec": _verifier_timeout_sec(ctx.config),
        "build_timeout_sec": 1800,
        "cpus": res["cpus"],
        "memory_mb": res["memory_mb"],
        "storage_mb": res["storage_mb"],
        "gpus": res["gpus"],
        "difficulty": ctx.config.get("difficulty", "hard"),
        "category": "ml-research",
        "tags": _domain_tags(ctx.task_id),
        "editable_files": _editable_files_view(effective_config),
        "extra_readable_files": _readable_only_files(effective_config),
        "visible_test_cmds": visible_test_cmds,
        "has_budget_check": _has_budget_check(task_dir),
        "budget_multiplier": _budget_multiplier(task_dir),
        "baseline_name": ctx.chosen_baseline or "noop",
        "env_vars": ctx.pkg_config.get("env") or {},
        "data_deps": ctx.pkg_config.get("data_deps") or [],
        "gpu_count": res["gpus"],
        "baseline_sections": baseline_sections,
        "read_sections": read_sections,
        "prompt_warnings": baseline_warnings + read_warnings,
        "oracle_cmd_overrides": ctx.oracle_cmd_overrides,
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
    scaffold_files = _stage_task_scaffold(mb, ctx, env_dir / "_scaffold")
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
    if gpus_int > 0:
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
    (sol_dir / "oracle_cmd_overrides.json").write_text(
        json.dumps(ctx.oracle_cmd_overrides, indent=2)
    )
    (sol_dir / "oracle_cmd_overrides.token").write_text(
        oracle_cmd_overrides_token + "\n"
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
    )
    (tests_dir / "meta" / "oracle_cmd_overrides.token").write_text(
        oracle_cmd_overrides_token + "\n"
    )

    # Atomic publish: rename `<task>.inprogress` → `<task>`. Any earlier raise
    # leaves only the staging dir, which the adapter's runner cleans up.
    out_dir.rename(final_dir)
    return final_dir


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

    # Cache target-file text per rel path so multiple ops on the same file
    # apply in sequence (matches native BaseAgent.apply_mid_edit semantics).
    file_texts: dict[str, list[str]] = {}

    def _load_target(rel: str) -> list[str] | None:
        if rel in file_texts:
            return file_texts[rel]
        pkg = _package_from_rel(ctx.config, ctx.package, rel)
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

    for pkg in _task_packages(ctx.config):
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
                ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
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
        created.append(_safe_rel_str(rel, field="mid_edit file"))
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
            new_text = "".join(_content_lines(content))
        elif kind == "replace" and target.exists():
            lines = target.read_text().splitlines(keepends=True)
            s = int(op["start_line"]) - 1
            e = _end_index(lines, int(op["end_line"]))
            new_text = "".join(lines[:s] + _content_lines(content) + lines[e:])
        elif kind == "insert":
            lines = target.read_text().splitlines(keepends=True) if target.exists() else []
            after = int(op.get("after_line", 0))
            new_text = "".join(lines[:after] + _content_lines(content) + lines[after:])
        elif kind == "delete" and target.exists():
            lines = target.read_text().splitlines(keepends=True)
            start, end = _delete_bounds(op)
            s = start - 1
            e = _end_index(lines, end)
            new_text = "".join(lines[:s]) + "".join(lines[e:])
        else:
            continue
        _assert_python_syntax(
            new_text,
            rel_path=rel,
            origin=f"pre_edit.py for {pkg}",
            task_id=f"materialize:{pkg}",
        )
        target.write_text(new_text)
    _PKG_SRC_CACHE[cache_key] = dst
    return dst


def _stage_verifier_assets(
    mb: MlsBenchRoot,
    ctx: TaskContext,
    tests_dir: Path,
    task_dir: Path,
    *,
    scaffold_dir: Path,
    config: dict | None = None,
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
    (meta / "config.json").write_text(
        json.dumps(config if config is not None else ctx.config, indent=2) + "\n"
    )
    if (task_dir / "budget_check.py").exists():
        shutil.copy2(task_dir / "budget_check.py", meta / "budget_check.py")
    (meta / "task_id").write_text(ctx.task_id + "\n")
    (meta / "package").write_text(ctx.package + "\n")
    (meta / "workdir").write_text(ctx.pkg_config.get("workdir", "/workspace") + "\n")
    (meta / "gpu_count").write_text(
        str(_resources(ctx.pkg_config, config if config is not None else ctx.config)["gpus"]) + "\n"
    )
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
        shutil.copytree(edits_src, meta / "edits", dirs_exist_ok=True)

    scripts_src = task_dir / "scripts"
    if scripts_src.exists():
        shutil.copytree(scripts_src, meta / "scripts", dirs_exist_ok=True)

    # Eval scripts.
    if scripts_src.exists():
        shutil.copytree(scripts_src, tests_dir / "eval" / "scripts", dirs_exist_ok=True)

    # Task-bundled third-party resources (e.g. dlm-dkv-policy ships pristine
    # snapshots of upstream dLLM-cache / D2Cache / Elastic-Cache LLaDA model
    # code under tasks/<t>/third_party/). These are loaded at verify time
    # via MLSBENCH_TASK_DIR / "third_party" / ..., which score_task.py points
    # at tests/meta/. Copy them into tests/meta/third_party/ so that lookup
    # resolves on Harbor — they're verifier-side resources, not editable
    # surface, so they belong here rather than in the base image.
    third_party_src = task_dir / "third_party"
    if third_party_src.exists():
        shutil.copytree(
            third_party_src, meta / "third_party",
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )

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

    declared_files = {
        _safe_rel_str(str(f["filename"]), field="files[].filename")
        for f in config.get("files", [])
        if f.get("filename")
    }

    seen_rel: set[str] = set()

    def _hash_bytes(b: bytes) -> str:
        return hashlib.sha256(b).hexdigest()

    def _emit(rel: str, content_bytes: bytes) -> None:
        if rel in seen_rel:
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

    for pkg in packages:
        pkg_src = _materialized_package_source(mb, pkg)
        if pkg_src is None:
            # Fail loudly: a missing pkg src means the manifest under-covers
            # that package; agent edits to non-declared files inside it
            # would not trigger a guard violation. Caller can decide whether
            # to abort. We raise so render fails fast.
            raise FileNotFoundError(
                f"_stage_pristine_assets: package source for {pkg!r} not found "
                f"under vendor/ or vendor/external_packages/; cannot produce a "
                f"complete pristine_manifest.json for task {ctx.task_id!r}. "
                f"Either fetch the package or remove it from this task's "
                f"test_cmds[].package list."
            )
        pkg_src_resolved = pkg_src.resolve()
        for fp in pkg_src.rglob("*"):
            # Use lstat to NOT follow symlinks — a vendored package might
            # contain a symlink pointing into /etc, /proc, or a giant data
            # blob; hashing the target would either leak or OOM render.
            try:
                st = fp.lstat()
            except OSError:
                continue
            import stat as _stat
            if not _stat.S_ISREG(st.st_mode):
                continue  # skip symlinks, sockets, FIFOs, block devices, etc.
            # Defensive: confirm the resolved path is still inside pkg_src.
            try:
                fp.resolve().relative_to(pkg_src_resolved)
            except ValueError:
                continue
            # Skip metadata noise that's never part of agent-visible workspace.
            if any(part in (".git", "__pycache__") for part in fp.relative_to(pkg_src).parts):
                continue
            if fp.suffix in (".pyc", ".pyo"):
                continue
            sub = fp.relative_to(pkg_src).as_posix()
            rel = f"{pkg}/{sub}" if sub else pkg
            # Scaffold overrides source on a per-file basis.
            scaffold_fp = scaffold_index.get(rel)
            if scaffold_fp is not None:
                _emit(rel, scaffold_fp.read_bytes())
            else:
                try:
                    _emit(rel, fp.read_bytes())
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
            if not exists:
                return None
            out = "".join(lines)
            _assert_python_syntax(
                out,
                rel_path=rel_filename,
                origin="mid_edit.py (and underlying pre_edit.py)",
                task_id=ctx.task_id,
            )
            return out
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
