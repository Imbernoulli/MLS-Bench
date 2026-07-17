#!/usr/bin/env python3
"""Harbor-side verifier for MLS-Bench tasks.

Runs three sub-commands inside the agent container:

    score_task.py guard       — edit-range diff guard
    score_task.py run-evals   — execute all eval scripts (visible + hidden)
    score_task.py score       — aggregate metrics → combined_score → reward.txt

Designed to be self-contained — only stdlib, plus mlsbench installed in the
image (via the prebuilt mlsbench/<pkg> base image).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
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

    If `pristine` doesn't exist, the agent created the file — caller decides
    whether to allow that based on `allow_create`.
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
    allow_create = bool(config.get("allow_create", False))
    # Optional per-task allowlist of workdir-relative prefixes dropped from the
    # guard — for image-baked data/build dirs inside the package dir that aren't
    # the agent's editable surface and were never in the render-time manifest
    # (issue #25.2 Error-1/Error-3).
    guard_exclude = tuple(config.get("guard_exclude", []) or [])

    workspace_files = _walk_workspace(workspace_root, guard_exclude)
    workspace_rel_strs = {p.as_posix() for p in workspace_files}

    # Guarded prefixes: every top-level dir referenced by editable list AND
    # the manifest (covers secondary packages even if no declared edits).
    guarded_prefixes = {Path(f).parts[0] for f in editable if f}
    guarded_prefixes |= {Path(f).parts[0] for f in manifest if f}

    # Disallowed creation: anything in workspace that is NOT in the manifest
    # (= agent created it post-start).
    if not allow_create:
        for rel in sorted(workspace_files):
            rel_str = rel.as_posix()
            if rel_str in manifest:
                continue
            violations.append(f"created new file (allow_create=false): {rel_str}")

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
            # Newly created file — handled by allow_create branch above.
            continue
        try:
            actual = hashlib.sha256(
                _safe_join(workspace_root, rel_str).read_bytes()
            ).hexdigest()
        except OSError:
            continue
        if actual != expected_sha:
            violations.append(f"{rel_str}: modified but not in editable file list")

    if violations:
        violation_out.parent.mkdir(parents=True, exist_ok=True)
        violation_out.write_text("\n".join(violations) + "\n")
        return 10
    return 0


# --------------------------------------------------------------------------- #
# Run all eval scripts (visible + hidden)
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
) -> dict[str, str]:
    env = os.environ.copy()
    default_pkg = _read_meta_text(task_meta, "package", "")
    package_envs = _load_package_envs(task_meta)
    pkg_name = str(tc.get("package") or default_pkg)
    for key, value in package_envs.get(pkg_name, package_envs.get(default_pkg, {})).items():
        if key == "HOME":
            env[key] = value
        else:
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
    env["MLSBENCH_TASK_DIR"] = str(task_meta)
    env["MLSBENCH_PKG_DIR"] = str(pkg_dir)
    env.setdefault("DATA_ROOT", "/data")
    env["MLSBENCH_LOCAL_PATH_MAP_JSON"] = json.dumps({
        "/workspace": str(workspace_root),
        f"/workspace/{pkg_name}": str(pkg_dir),
        "/data": "/data",
    })
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


def _infer_reserved_gpu_count(config: dict) -> int:
    if config.get("use_cuda") is False:
        return 0
    test_cmds = list(config.get("test_cmds", []) or [])
    if not config.get("use_cuda") and not any("compute" in tc for tc in test_cmds):
        return 0

    peak_gpus = 0
    n_seeds = max(1, len(_config_seeds(config)))
    for entries in _group_entries(test_cmds).values():
        whole = 0
        fractionals: list[float] = []
        for _, tc in entries:
            compute = _test_cmd_compute(tc)
            if compute >= 1.0:
                whole += n_seeds * max(1, math.ceil(compute))
            elif compute > 0.0:
                fractionals.extend([compute] * n_seeds)
        peak_gpus = max(peak_gpus, whole + _bin_pack_fractional_gpus(fractionals))
    return max(1, peak_gpus) if peak_gpus else 0


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


def _copy_task_meta_for_budget(
    task_meta: Path, scratch_dir: Path,
    effective_test_cmds: list[dict] | None = None,
) -> None:
    scratch_dir.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "budget_check.py"):
        src = task_meta / name
        if src.exists():
            shutil.copy2(src, scratch_dir / name)
    for name in ("edits", "scripts"):
        src = task_meta / name
        if src.exists():
            shutil.copytree(src, scratch_dir / name, dirs_exist_ok=True)
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


def _build_eval_task_dir(task_meta: Path) -> Path:
    """Throwaway dir exposing ONLY the eval-time resources (scripts/ data/
    third_party/) that some eval wrappers reach via /workspace/_task. It
    deliberately EXCLUDES the scoring metadata (parser.py / score_spec.py /
    config.json / leaderboard.csv) that cmd_score imports: eval runs agent-
    authored package code as root, so any path it can reach is writable (chmod
    a-w doesn't stop root), and exposing the real task_meta would let a
    submission overwrite the parser/spec before the score phase. The real
    task_meta stays at its unexposed random /tmp path."""
    d = Path(tempfile.mkdtemp(prefix="mlsbench-evaltask-"))
    for sub in ("scripts", "data", "third_party"):
        src = task_meta / sub
        if src.exists():
            shutil.copytree(src, d / sub, dirs_exist_ok=True)
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
            proc = subprocess.run(
                [python_bin, str(scratch_dir / "budget_check.py")],
                cwd=str(pkg_dir),
                env=budget_env,
                stdout=fh,
                stderr=subprocess.STDOUT,
                timeout=120,
                check=False,
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
    try:
        state["fh"].close()
    except OSError:
        pass
    return {
        "seed": seed,
        "rc": rc,
        "log": str(state["log_path"]),
        "elapsed": elapsed,
    }


def _run_eval_wave(
    *,
    tasks: list[dict],
    assignments: list[str | None],
    task_meta: Path,
    workspace_root: Path,
    default_pkg: str,
    out_dir: Path,
) -> dict[tuple[int, int], dict]:
    timeout_secs = max(
        _parse_time_to_seconds(task["entry"]["tc"].get("time", "1:00:00"))
        for task in tasks
    ) + 300
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
        )
        if gpu_devices:
            env["CUDA_VISIBLE_DEVICES"] = gpu_devices
            env["NVIDIA_VISIBLE_DEVICES"] = gpu_devices
        fh = log_path.open("w")
        t_start = time.time()
        try:
            proc = subprocess.Popen(
                ["bash", str(entry["script"])],
                cwd=str(pkg_dir),
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception as exc:
            fh.write(f"[ERROR] failed to start eval command: {exc}\n")
            fh.close()
            results[(entry["idx"], seed)] = {
                "seed": seed,
                "rc": 127,
                "log": str(log_path),
                "elapsed": time.time() - t_start,
            }
            continue
        running.append({
            "entry": entry,
            "seed": seed,
            "proc": proc,
            "fh": fh,
            "start": t_start,
            "log_path": log_path,
        })

    while running and time.time() < deadline:
        still_running: list[dict] = []
        for state in running:
            rc = state["proc"].poll()
            if rc is None:
                still_running.append(state)
            else:
                results[(state["entry"]["idx"], state["seed"])] = _finish_process_record(
                    state,
                    state["seed"],
                    rc,
                )
        running = still_running
        if running:
            time.sleep(0.5)

    for state in running:
        try:
            state["fh"].write(
                f"\n[TIMEOUT] Command timed out after {timeout_secs} seconds.\n"
            )
            state["fh"].flush()
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


def _parse_oracle_cmd_overrides(raw: str | None) -> list[dict[str, str]]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid oracle cmd overrides JSON: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError("oracle cmd overrides must be a JSON list")

    out: list[dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("oracle cmd override entries must be objects")
        cmd = str(item.get("cmd", "")).strip()
        if not cmd:
            raise ValueError("oracle cmd override missing non-empty cmd")
        if "labels" in item:
            labels = item.get("labels") or [""]
            if not isinstance(labels, list):
                labels = [labels]
            out.extend({"label": str(label), "cmd": cmd} for label in labels)
        else:
            out.append({"label": str(item.get("label", "")), "cmd": cmd})
    return out


def _apply_oracle_cmd_overrides(
    test_cmds: list[dict],
    overrides: list[dict[str, str]],
) -> list[dict]:
    result = [dict(tc) for tc in test_cmds]
    for override in overrides:
        label = override["label"]
        for entry in result:
            if not label or str(entry.get("label", "")) == label:
                entry["cmd"] = override["cmd"]
    return result


def cmd_run_evals(args: argparse.Namespace) -> int:
    task_meta = Path(args.task_meta)
    workspace_root = Path(args.workspace)
    default_pkg = _read_meta_text(task_meta, "package", "")
    eval_root = Path(args.eval_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = _load_task_config(task_meta)
    test_cmds = list(config.get("test_cmds", []))
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
                runnable_tasks.append(task)
                runnable_assignments.append(gpu_devices)

            if not runnable_tasks:
                continue

            # Expose the legacy /workspace/_task path that some eval wrappers
            # reference (humanoid: `python _task/scripts/...`; dllm:
            # `--data-path /workspace/_task/data/...`). Native MLSBench bind-mounts
            # the task dir there, but Harbor has no bind mounts. We point _task at
            # a MINIMAL throwaway copy (scripts/data/third_party only) rather than
            # task_meta, so eval-time agent code (running as root) cannot reach and
            # overwrite the scoring metadata (parser.py/score_spec.py/config.json)
            # before the score phase. The guard exempts the _task top-level dir and
            # runs as a separate pre-eval invocation, so this never affects the
            # diff. (issue #25.4, #25.5; Codex P1 on #29)
            eval_task_dir = _build_eval_task_dir(task_meta)
            eval_task_links = _install_budget_legacy_links(eval_task_dir, workspace_root)
            try:
                wave_results = _run_eval_wave(
                    tasks=runnable_tasks,
                    assignments=runnable_assignments,
                    task_meta=task_meta,
                    workspace_root=workspace_root,
                    default_pkg=default_pkg,
                    out_dir=out_dir,
                )
            finally:
                _remove_budget_legacy_links(eval_task_links)
                shutil.rmtree(eval_task_dir, ignore_errors=True)
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

    for idx, _tc in enumerate(test_cmds):
        for seed in seeds:
            record = records.get((idx, seed))
            if record is not None:
                summary[idx]["logs"].append(record)

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
        import mlsbench.agent.parsers  # ensures the task parser inherits this version

        import importlib.util
        parser_spec = importlib.util.spec_from_file_location(
            "task_parser", task_meta / "parser.py"
        )
        task_parser = importlib.util.module_from_spec(parser_spec)
        parser_spec.loader.exec_module(task_parser)
    except Exception as exc:
        reward_out.write_text("0\n")
        (out_dir / "score_error.txt").write_text(f"import failed: {exc}\n")
        return 0

    config = json.loads((task_meta / "config.json").read_text())
    summary_path = out_dir / "eval_summary.json"
    if not summary_path.exists():
        reward_out.write_text("0\n")
        (out_dir / "score_error.txt").write_text("eval_summary.json missing\n")
        return 0
    summary = json.loads(summary_path.read_text())

    # Parse every log, aggregate per-seed metrics, then mean across seeds.
    test_cmd_by_label = {tc.get("label", tc["cmd"]): tc for tc in config.get("test_cmds", [])}
    per_seed_metrics: dict[int, dict] = {}
    for entry in summary:
        label = entry["label"]
        tc = test_cmd_by_label.get(label)
        if tc is None:
            continue
        # Every MLS-Bench parser.py defines `class Parser(OutputParser)` with
        # `parse(self, cmd_label, raw_output) -> ParseResult`.
        parser_inst = task_parser.Parser()
        for log_info in entry.get("logs", []):
            if "log" not in log_info:
                continue
            seed = int(log_info["seed"])
            log_path = Path(log_info["log"])
            if not log_path.exists():
                continue
            log_text = log_path.read_text()
            try:
                parsed = parser_inst.parse(label, log_text)
            except Exception as exc:
                with (out_dir / "parse_errors.txt").open("a") as fh:
                    fh.write(f"{label} seed {seed}: {exc}\n")
                continue
            metrics = getattr(parsed, "metrics", None) or {}
            seed_metrics = per_seed_metrics.setdefault(seed, {})
            seed_metrics.update(metrics)
            if "elapsed" in log_info:
                try:
                    seed_metrics[f"elapsed_{label}"] = float(log_info["elapsed"])
                except (TypeError, ValueError):
                    pass

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
    }, indent=2))
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
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

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
