#!/usr/bin/env python3
"""Harbor-side verifier for MLS-Bench tasks.

Runs three sub-commands inside the agent container:

    score_task.py guard       — edit-range diff guard
    score_task.py run-evals   — execute all configured eval scripts
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


_FAILURE_MARKER_RE = re.compile(
    r"(?m)^\s*("
    r"(?:[A-Z][A-Z0-9_]*(?:FALLBACK|NONFINITE)[A-Z0-9_]*)"
    r"|SURFACE_ERROR"
    r"|TRAIN_ERROR"
    r"|EVAL_FAILED"
    r"|PROMPT_CFG\s+build_prompt\s+failed"
    r"|TOKENSTRAT_CFG\s+(?:model_cfg_)?set_failed"
    r"|LAYER_CFG\s+surgery_failed"
    r"|PROMPT_TEMPLATE_ERROR"
    r"|TOKEN_SURFACE_ERROR"
    r"|DETECTOR_ERROR"
    r")\b"
)

_STANDARD_FAILURE_MARKER_RES = (
    re.compile(
        r"(?i)\b("
        r"(?:verification|evaluation|training)\s+(?:has\s+)?failed"
        r"|(?:verification|evaluation|training)\s+did\s+not\s+complete"
        r")\b"
    ),
    re.compile(r"(?m)^\s*(Traceback\s+\(most recent call last\):)"),
    re.compile(
        r"(?mi)^\s*"
        r"(?:(?:"
        r"\d{4}-\d{2}-\d{2}(?:[T ][^\s]+)?"
        r"|\[[0-9][^\]\r\n]*\]"
        r")\s+)?"
        r"(?:\[(?:ERROR|FATAL|CRITICAL)\]|(?:ERROR|FATAL|CRITICAL)\b)"
        r"\s*:?\s+("
        r"(?:[A-Za-z_][A-Za-z0-9_]*\.)*"
        r"[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)"
        r")\s*:"
    ),
)


def _failure_marker(raw_output: str) -> str | None:
    match = _FAILURE_MARKER_RE.search(raw_output)
    if match:
        return match.group(1)
    for pattern in _STANDARD_FAILURE_MARKER_RES:
        match = pattern.search(raw_output)
        if match:
            return match.group(1)
    return None


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
            range(r.start, r.end + 1)
            if r.start != -1 and r.end != -1
            else range(1, len(pristine_lines) + 1)
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


def _walk_workspace(workspace_root: Path) -> set[Path]:
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
        out.add(p.relative_to(workspace_root))
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

    workspace_files = _walk_workspace(workspace_root)
    workspace_rel_strs = {p.as_posix() for p in workspace_files}

    # Guarded prefixes: every top-level dir referenced by editable list AND
    # the manifest (covers secondary packages even if no declared edits).
    guarded_prefixes = {Path(f).parts[0] for f in editable if f}
    guarded_prefixes |= {Path(f).parts[0] for f in manifest if f}

    # Disallowed creation: anything in workspace under a guarded prefix that
    # is NOT in the manifest (= agent created it post-start).
    if not allow_create:
        for rel in sorted(workspace_files):
            if not rel.parts or rel.parts[0] not in guarded_prefixes:
                continue
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
# Run all configured eval scripts
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


def _install_verifier_package_files(task_meta: Path, workspace_root: Path) -> None:
    src_root = task_meta / "verifier_package_files"
    if not src_root.is_dir():
        return
    for src in sorted(src_root.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(src_root).as_posix()
        dst = _safe_join(workspace_root, rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _normalize_pkg_name(name: str) -> str:
    return str(name).lower().replace("-", "").replace("_", "")


def _eval_env(
    *,
    task_meta: Path,
    eval_task_meta: Path,
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
    artifact_root_raw = os.environ.get("MLSBENCH_EVAL_ARTIFACT_ROOT")
    artifact_root = Path(artifact_root_raw) if artifact_root_raw else out_dir
    save_path = artifact_root / "save"
    output_dir = save_path / task_id / "harbor" / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_home = artifact_root / "eval-home" / f"seed_{seed}"
    eval_home.mkdir(parents=True, exist_ok=True)
    eval_uid = os.environ.get("MLSBENCH_EVAL_UID")
    eval_gid = os.environ.get("MLSBENCH_EVAL_GID")
    if eval_uid is not None and eval_gid is not None and os.geteuid() == 0:
        uid, gid = int(eval_uid), int(eval_gid)
        for writable in (save_path, output_dir, eval_home):
            os.chown(writable, uid, gid)

    env["SAVE_PATH"] = str(save_path)
    env["OUTPUT_DIR"] = str(output_dir)
    env["HOME"] = str(eval_home)
    env["XDG_CACHE_HOME"] = str(eval_home / ".cache")
    env["SEED"] = str(seed)
    label = str(tc.get("label", ""))
    if label:
        env["ENV"] = label
    env["MLSBENCH_TASK_DIR"] = str(eval_task_meta)
    env["MLSBENCH_PKG_DIR"] = str(pkg_dir)
    # Task-specific verifier data is copied into the sanitized runtime tree by
    # test.sh. Point scripts at that exact tree; never fall back to image-baked
    # /data paths, which are deliberately pruned for these tasks.
    env["MLSBENCH_VERIFIER_DATA_ROOT"] = str(eval_task_meta / "data")
    env.setdefault("DATA_ROOT", "/data")
    env["MLSBENCH_LOCAL_PATH_MAP_JSON"] = json.dumps({
        "/workspace": str(workspace_root),
        f"/workspace/{pkg_name}": str(pkg_dir),
        "/data": "/data",
    })
    return env


def _eval_preexec_fn():
    """Drop eval commands to an unprivileged uid with no privilege regain."""
    raw_uid = os.environ.get("MLSBENCH_EVAL_UID")
    raw_gid = os.environ.get("MLSBENCH_EVAL_GID")
    if raw_uid is None or raw_gid is None or os.geteuid() != 0:
        return None
    uid, gid = int(raw_uid), int(raw_gid)

    def drop() -> None:
        try:
            import ctypes

            libc = ctypes.CDLL(None, use_errno=True)
            if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
                raise OSError(ctypes.get_errno(), "prctl(PR_SET_NO_NEW_PRIVS) failed")
        except ImportError:
            pass
        os.setgroups([])
        os.setgid(gid)
        os.setuid(uid)
        os.umask(0o077)

    return drop


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


def _infer_reserved_gpu_count(config: dict) -> int:
    if config.get("use_cuda") is False:
        return 0
    test_cmds = list(config.get("test_cmds", []) or [])
    if not config.get("use_cuda") and not any("compute" in tc for tc in test_cmds):
        return 0

    peak_gpus = 0
    n_seeds = max(1, len(_config_seeds(config)))
    for entries in _group_entries(test_cmds).values():
        whole_per_seed = 0
        fractional_per_seed = 0.0
        for _, tc in entries:
            compute = _test_cmd_compute(tc)
            if compute >= 1.0:
                whole_per_seed += max(1, math.ceil(compute))
            elif compute > 0.0:
                fractional_per_seed += compute
        total_whole = n_seeds * whole_per_seed
        total_fractional = n_seeds * fractional_per_seed
        peak_gpus = max(peak_gpus, total_whole + max(0, math.ceil(total_fractional)))
    return max(1, peak_gpus) if peak_gpus else 0


def _reserved_gpu_count(task_meta: Path, config: dict) -> int:
    p = task_meta / "gpu_count"
    if p.exists():
        try:
            return max(0, int(p.read_text().strip() or "0"))
        except ValueError:
            pass
    return _infer_reserved_gpu_count(config)


def _gpu_compute_cap(task_meta: Path) -> int:
    """Per-GPU compute capability relative to H100.

    B200 ≈ 2× H100, so gpu_compute_cap=2 means each physical GPU
    satisfies 2 units of the original H100-based ``compute`` field.
    Defaults to 1 (H100 parity).
    """
    p = task_meta / "gpu_compute_cap"
    if p.exists():
        try:
            return max(1, int(p.read_text().strip() or "1"))
        except ValueError:
            pass
    return 1


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


def _task_gpu_need(task: dict, gpu_cap: int = 1) -> int:
    compute = _test_cmd_compute(task["entry"]["tc"])
    if compute <= 0.0:
        return 0
    # Scale by per-GPU capability (e.g. B200 ≈ 2× H100 → gpu_cap=2).
    effective = compute / gpu_cap
    if effective >= 1.0:
        return max(1, math.ceil(effective))
    return 1


def _try_allocate_task_to_remaining(
    task: dict,
    remaining: dict[str, float],
    gpu_cap: int = 1,
) -> str | None:
    compute = _test_cmd_compute(task["entry"]["tc"])
    if compute <= 0.0:
        return None
    effective = compute / gpu_cap
    if effective >= 1.0:
        need = max(1, math.ceil(effective))
        free = [device for device, cap in remaining.items() if cap >= 1.0]
        if len(free) < need:
            return None
        chosen = free[:need]
        for device in chosen:
            remaining[device] = 0.0
        return ",".join(chosen)

    chosen = next((device for device, cap in remaining.items() if cap >= effective), None)
    if chosen is None:
        return None
    remaining[chosen] -= effective
    return chosen


def _allocate_group_gpu_assignments(
    tasks: list[dict],
    devices: list[str],
    gpu_cap: int = 1,
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
        assignment = _try_allocate_task_to_remaining(task, remaining, gpu_cap)
        if assignment is None and _task_gpu_need(task, gpu_cap) > 0:
            return None
        assignments[idx] = assignment
    return assignments


def _partition_group_gpu_batches(
    tasks: list[dict],
    devices: list[str],
    gpu_cap: int = 1,
    serial: bool = False,
) -> list[tuple[list[dict], list[str | None]]] | None:
    if serial:
        batches: list[tuple[list[dict], list[str | None]]] = []
        for task in tasks:
            assignments = _allocate_group_gpu_assignments([task], devices, gpu_cap)
            if assignments is None:
                return None
            batches.append(([task], assignments))
        return batches

    if not devices:
        return [(list(tasks), [None] * len(tasks))]

    batches: list[tuple[list[dict], list[str | None]]] = []
    current: list[dict] = []
    for task in tasks:
        trial = [*current, task]
        if _allocate_group_gpu_assignments(trial, devices, gpu_cap) is None:
            if not current:
                return None
            assignments = _allocate_group_gpu_assignments(current, devices, gpu_cap)
            if assignments is None:
                return None
            batches.append((current, assignments))
            current = [task]
            if _allocate_group_gpu_assignments(current, devices, gpu_cap) is None:
                return None
        else:
            current = trial

    if current:
        assignments = _allocate_group_gpu_assignments(current, devices, gpu_cap)
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
    task_meta: Path,
    scratch_dir: Path,
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


def _install_task_meta_legacy_links(task_meta: Path, workspace_root: Path) -> None:
    for dst in {workspace_root / "_task", Path("/workspace/_task")}:
        try:
            if dst.exists() or dst.is_symlink():
                if dst.is_dir() and not dst.is_symlink():
                    shutil.rmtree(dst)
                else:
                    dst.unlink()
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(task_meta, dst, target_is_directory=True)
        except OSError:
            continue


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
    budget_timeout = int(os.environ.get("MLSBENCH_BUDGET_TIMEOUT_SEC", "600"))
    legacy_links: list[Path] = []
    with log_path.open("w") as fh:
        try:
            _copy_task_meta_for_budget(task_meta, scratch_dir, effective_test_cmds)
            if os.environ.get("MLSBENCH_EVAL_UID") is not None and os.geteuid() == 0:
                os.chown(
                    scratch_dir,
                    int(os.environ["MLSBENCH_EVAL_UID"]),
                    int(os.environ["MLSBENCH_EVAL_GID"]),
                )
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
                timeout=budget_timeout,
                check=False,
                preexec_fn=_eval_preexec_fn(),
            )
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            fh.write(
                f"\n[BUDGET CHECK TIMEOUT] budget_check.py took "
                f">{budget_timeout}s\n"
            )
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
    if os.environ.get("MLSBENCH_CLEAN_PROCESS_GROUPS") == "1":
        _kill_process_group(state["proc"].pid, timeout=5.0)
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
    eval_task_meta: Path,
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

    # budget_check.py may temporarily replace /workspace/_task with a scratch
    # copy and remove it afterwards. Restore the hidden verifier meta link
    # before launching eval scripts that resolve _task/scripts or _task/data.
    _install_task_meta_legacy_links(eval_task_meta, workspace_root)

    for task, gpu_devices in zip(tasks, assignments):
        entry = task["entry"]
        seed = int(task["seed"])
        log_path = _eval_log_path(out_dir, entry["label"], seed)
        pkg_dir = _package_dir(workspace_root, default_pkg, entry["tc"])
        env = _eval_env(
            task_meta=task_meta,
            eval_task_meta=eval_task_meta,
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
                preexec_fn=_eval_preexec_fn(),
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
    eval_task_meta = Path(getattr(args, "eval_task_meta", None) or args.task_meta)
    workspace_root = Path(args.workspace)
    default_pkg = _read_meta_text(task_meta, "package", "")
    eval_root = Path(args.eval_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "eval_summary.json",
        "metrics.json",
        "verification_result.json",
        "score_error.txt",
        "parse_errors.txt",
        "budget_violation.txt",
    ):
        (out_dir / name).unlink(missing_ok=True)
    for stale_log in out_dir.glob("*.log"):
        stale_log.unlink(missing_ok=True)

    config = _load_task_config(task_meta)
    try:
        _install_verifier_package_files(task_meta, workspace_root)
    except Exception as exc:
        (out_dir / "score_error.txt").write_text(
            f"failed to install verifier-only package files: {exc}\n"
        )
        return 125
    test_cmds = list(config.get("test_cmds", []))
    oracle_cmd_overrides = _parse_oracle_cmd_overrides(
        getattr(args, "oracle_cmd_overrides", None)
    )
    if oracle_cmd_overrides:
        test_cmds = _apply_oracle_cmd_overrides(test_cmds, oracle_cmd_overrides)
    seeds = _config_seeds(config)

    summary = [
        {"label": tc.get("label", tc.get("cmd", "test")), "logs": []}
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
    gpu_cap = _gpu_compute_cap(task_meta)
    serial_evals = config.get("_verifier_serial") is True

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
            need = _task_gpu_need(task, gpu_cap)
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

        batches = _partition_group_gpu_batches(
            schedulable,
            devices,
            gpu_cap,
            serial=serial_evals,
        )
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
                    eval_task_meta=eval_task_meta,
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

            wave_results = _run_eval_wave(
                tasks=runnable_tasks,
                assignments=runnable_assignments,
                task_meta=task_meta,
                eval_task_meta=eval_task_meta,
                workspace_root=workspace_root,
                default_pkg=default_pkg,
                out_dir=out_dir,
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

    for idx, _tc in enumerate(test_cmds):
        for seed in seeds:
            record = records.get((idx, seed))
            if record is not None:
                summary[idx]["logs"].append(record)

    (out_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2))

    failed = []
    for entry in summary:
        label = entry["label"]
        logs_by_seed = {int(log["seed"]): log for log in entry.get("logs", []) if "seed" in log}
        for seed in seeds:
            log = logs_by_seed.get(int(seed))
            if log is None:
                failed.append(f"{label} seed {seed}: missing eval result")
                continue
            try:
                rc = int(log.get("rc", 125))
            except (TypeError, ValueError):
                rc = 125
            if rc != 0:
                failed.append(f"{label} seed {seed}: eval exited with rc={rc}")
                continue
            log_path = Path(str(log.get("log", "")))
            if not log_path.is_file():
                failed.append(f"{label} seed {seed}: eval log missing")
                continue
            raw_log = log_path.read_text(errors="replace")
            if not raw_log.strip():
                failed.append(f"{label} seed {seed}: eval log is empty")
                continue
            marker = _failure_marker(raw_log)
            if marker:
                failed.append(f"{label} seed {seed}: harness failure marker {marker}")

    if failed:
        (out_dir / "score_error.txt").write_text(
            "one or more required evaluations failed; reward forced to 0:\n"
            + "\n".join(failed)
            + "\n"
        )
        return 1
    return 0


# --------------------------------------------------------------------------- #
# Score: parse logs, aggregate, write reward
# --------------------------------------------------------------------------- #

def _aggregate_metrics(metrics_list: list[dict]) -> dict:
    if not metrics_list:
        return {}
    if any(_parser_metrics_error(metrics) for metrics in metrics_list):
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


def _parser_metrics_error(metrics: object) -> str | None:
    """Return why a parser metric mapping is unsafe, or ``None``."""
    if not isinstance(metrics, dict):
        return f"expected-dict-got-{type(metrics).__name__}"
    for key, value in metrics.items():
        if not isinstance(key, str) or not key:
            return f"invalid-key-{key!r}"
        if isinstance(value, bool):
            return f"{key}=bool"
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError):
            return f"{key}=nonnumeric"
        if not math.isfinite(numeric):
            return f"{key}=nonfinite"
    return None


def _duplicate_authoritative_metric_lines(
    parser_type,
    cmd_label: str,
    raw_output: str,
    score_metric_keys: set[str],
) -> dict[str, list[int]]:
    """Find scored metric keys independently emitted by multiple log lines.

    Task parsers commonly scan the whole log and assign into a dict, which
    silently makes the last matching line authoritative. Editable solution code
    runs in the harness process and can print a forged matching record from an
    ``atexit`` callback after the harness emits its real result. Re-parse each
    line in isolation so a second record for the same scored key is ambiguous
    and therefore fails closed.

    A single line may legitimately emit several distinct scored metrics. Lines
    that do not independently parse, or that only emit diagnostic metrics, do
    not participate in this check.
    """
    occurrences: dict[str, list[int]] = {}
    line_parser = parser_type()
    for line_number, line in enumerate(raw_output.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            parsed = line_parser.parse(cmd_label, line)
        except Exception:
            continue
        metrics = getattr(parsed, "metrics", None)
        if _parser_metrics_error(metrics) is not None:
            continue
        for key in set(metrics) & score_metric_keys:
            occurrences.setdefault(key, []).append(line_number)
    return {
        key: line_numbers
        for key, line_numbers in occurrences.items()
        if len(line_numbers) > 1
    }


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


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        tmp.write_text(content)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _canonical_json_text(payload: object) -> str:
    """Serialize JSON exactly as Mangrove's artifact transport normalizer does."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _validate_eval_summary(summary: object, config: dict) -> str | None:
    """Require every configured setting and seed to have one successful log."""
    if not isinstance(summary, list):
        return "eval_summary.json must contain a list"

    test_cmds = list(config.get("test_cmds", []))
    seeds = _config_seeds(config)
    if not test_cmds:
        return "config declares no test_cmds"

    expected_labels = [
        str(tc.get("label", tc.get("cmd", "test")))
        for tc in test_cmds
    ]
    duplicate_labels = sorted({label for label in expected_labels if expected_labels.count(label) > 1})
    if duplicate_labels:
        return f"config declares duplicate test labels: {', '.join(duplicate_labels)}"
    if len(set(seeds)) != len(seeds):
        return "config declares duplicate seeds"

    entries_by_label: dict[str, list[dict]] = {}
    for entry in summary:
        if not isinstance(entry, dict):
            return "eval_summary.json contains a non-object entry"
        label = str(entry.get("label", ""))
        entries_by_label.setdefault(label, []).append(entry)

    failures: list[str] = []
    unexpected_labels = sorted(set(entries_by_label) - set(expected_labels))
    if unexpected_labels:
        failures.append(f"unexpected summary labels: {', '.join(unexpected_labels)}")
    for tc in test_cmds:
        label = str(tc.get("label", tc.get("cmd", "test")))
        matching_entries = entries_by_label.get(label, [])
        if len(matching_entries) != 1:
            failures.append(
                f"{label}: expected exactly one summary entry, found {len(matching_entries)}"
            )
            continue

        logs_by_seed: dict[int, list[dict]] = {}
        logs = matching_entries[0].get("logs", [])
        if not isinstance(logs, list):
            failures.append(f"{label}: logs must be a list")
            continue
        for log in logs:
            if not isinstance(log, dict) or "seed" not in log:
                failures.append(f"{label}: malformed log record")
                continue
            try:
                seed = int(log["seed"])
            except (TypeError, ValueError):
                failures.append(f"{label}: invalid seed {log.get('seed')!r}")
                continue
            logs_by_seed.setdefault(seed, []).append(log)

        unexpected_seeds = sorted(set(logs_by_seed) - {int(seed) for seed in seeds})
        if unexpected_seeds:
            failures.append(
                f"{label}: unexpected seeds {', '.join(str(seed) for seed in unexpected_seeds)}"
            )

        for seed in seeds:
            seed_logs = logs_by_seed.get(int(seed), [])
            if len(seed_logs) != 1:
                failures.append(
                    f"{label} seed {seed}: expected exactly one log, found {len(seed_logs)}"
                )
                continue
            log = seed_logs[0]
            rc = log.get("rc")
            if type(rc) is not int or rc != 0:
                failures.append(
                    f"{label} seed {seed}: eval rc must be integer 0, got {rc!r}"
                )
            log_path = log.get("log")
            if not log_path or not Path(str(log_path)).is_file():
                failures.append(f"{label} seed {seed}: eval log missing")
                continue
            raw_log = Path(str(log_path)).read_text(errors="replace")
            if not raw_log.strip():
                failures.append(f"{label} seed {seed}: eval log is empty")
                continue
            marker = _failure_marker(raw_log)
            if marker:
                failures.append(f"{label} seed {seed}: harness failure marker {marker}")

    return "; ".join(failures) if failures else None

def cmd_score(args: argparse.Namespace) -> int:
    task_meta = Path(args.task_meta)
    out_dir = Path(args.out_dir)
    reward_out = Path(args.reward_out)
    reward_out.parent.mkdir(parents=True, exist_ok=True)
    # Score can be invoked independently during debugging, not only after
    # run-evals. Invalidate every previous success artifact before importing
    # task code or parsing a new log matrix.
    _atomic_write_text(reward_out, "0\n")
    for stale_name in ("metrics.json", "verification_result.json"):
        (out_dir / stale_name).unlink(missing_ok=True)

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
        from mlsbench.scoring.evaluate import (  # type: ignore[import-not-found]
            load_expanded_spec,
            score_record,
            score_record_details,
        )
        from mlsbench.scoring.anchors import BaselineAnchors  # type: ignore[import-not-found]
        from mlsbench.scoring.spec import validate_score_spec  # type: ignore[import-not-found]
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

    try:
        config = json.loads((task_meta / "config.json").read_text())
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        reward_out.write_text("0\n")
        (out_dir / "score_error.txt").write_text(f"invalid config.json: {exc}\n")
        return 0
    summary_path = out_dir / "eval_summary.json"
    if not summary_path.exists():
        reward_out.write_text("0\n")
        (out_dir / "score_error.txt").write_text("eval_summary.json missing\n")
        return 0
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        reward_out.write_text("0\n")
        (out_dir / "score_error.txt").write_text(f"invalid eval_summary.json: {exc}\n")
        return 0

    summary_error = _validate_eval_summary(summary, config)
    if summary_error:
        reward_out.write_text("0\n")
        (out_dir / "score_error.txt").write_text(
            f"required evaluation did not complete successfully: {summary_error}\n"
        )
        return 0

    try:
        anchors = BaselineAnchors(task_meta)
        spec = load_expanded_spec(task_meta, anchors)
    except Exception as exc:
        reward_out.write_text("0\n")
        (out_dir / "score_error.txt").write_text(
            f"score spec load failed: {type(exc).__name__}: {exc}\n"
        )
        return 0
    if spec is None:
        reward_out.write_text("0\n")
        (out_dir / "score_error.txt").write_text("score_spec missing or invalid\n")
        return 0
    spec_errors = validate_score_spec(
        spec,
        [
            term.metric
            for term in spec.terms.values()
            if term.role != "drop" and isinstance(term.metric, str)
        ],
    )
    if spec_errors:
        reward_out.write_text("0\n")
        (out_dir / "score_error.txt").write_text(
            "invalid score specification; reward forced to 0:\n"
            + "\n".join(spec_errors)
            + "\n"
        )
        return 0

    setting_metric_keys: dict[str, set[str]] = {}
    for setting_name, setting_spec in spec.settings.items():
        keys: set[str] = set()
        for term_name, _weight in setting_spec.terms:
            term_spec = spec.terms.get(term_name)
            if term_spec is not None:
                keys.add(term_spec.metric)
        for term_name in setting_spec.constraints:
            term_spec = spec.terms.get(term_name)
            if term_spec is not None:
                keys.add(term_spec.metric)
        if not keys:
            reward_out.write_text("0\n")
            (out_dir / "score_error.txt").write_text(
                f"score setting {setting_name!r} declares no metrics\n"
            )
            return 0
        setting_metric_keys[str(setting_name)] = keys
    all_score_metrics = set().union(*setting_metric_keys.values())

    # Parse every log. A command may contribute only its declared/inferred
    # score setting, preventing metrics from one successful log from filling a
    # missing sibling setting or seed.
    test_cmd_by_label = {tc.get("label", tc["cmd"]): tc for tc in config.get("test_cmds", [])}
    per_seed_metrics: dict[int, dict] = {}
    seen_seed_settings: dict[int, set[str]] = {}
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
                reward_out.write_text("0\n")
                (out_dir / "score_error.txt").write_text(
                    f"parser failed for {label} seed {seed}; reward forced to 0\n"
                )
                return 0
            metrics = getattr(parsed, "metrics", None)
            if metrics is None:
                metrics = {}
            parser_metric_error = _parser_metrics_error(metrics)
            if parser_metric_error is not None:
                reward_out.write_text("0\n")
                (out_dir / "score_error.txt").write_text(
                    f"invalid parser metric for {label} seed {seed}: "
                    f"{parser_metric_error}; reward forced to 0\n"
                )
                return 0
            duplicate_metric_lines = _duplicate_authoritative_metric_lines(
                task_parser.Parser,
                label,
                log_text,
                all_score_metrics,
            )
            if duplicate_metric_lines:
                details = ", ".join(
                    f"{key} on lines {','.join(str(line) for line in lines)}"
                    for key, lines in sorted(duplicate_metric_lines.items())
                )
                reward_out.write_text("0\n")
                (out_dir / "score_error.txt").write_text(
                    f"duplicate authoritative metric for {label} seed {seed}: "
                    f"{details}; reward forced to 0\n"
                )
                return 0
            if not _has_real_metrics(metrics):
                reward_out.write_text("0\n")
                (out_dir / "score_error.txt").write_text(
                    f"no metrics extracted for {label} seed {seed}; reward forced to 0\n"
                )
                return 0
            declared_settings = tc.get("score_settings")
            if declared_settings is None:
                covered_settings = [
                    setting_name
                    for setting_name, required in setting_metric_keys.items()
                    if required.issubset(metrics)
                ]
                single_command_covers_matrix = (
                    len(test_cmd_by_label) == 1
                    and set(covered_settings) == set(setting_metric_keys)
                )
                if len(covered_settings) != 1 and not single_command_covers_matrix:
                    reward_out.write_text("0\n")
                    (out_dir / "score_error.txt").write_text(
                        f"{label} seed {seed} must cover exactly one score setting; "
                        f"covered={covered_settings}\n"
                    )
                    return 0
            else:
                if not isinstance(declared_settings, list) or not declared_settings:
                    reward_out.write_text("0\n")
                    (out_dir / "score_error.txt").write_text(
                        f"{label} has invalid score_settings declaration\n"
                    )
                    return 0
                covered_settings = [str(name) for name in declared_settings]
                unknown = sorted(set(covered_settings) - set(setting_metric_keys))
                if unknown:
                    reward_out.write_text("0\n")
                    (out_dir / "score_error.txt").write_text(
                        f"{label} declares unknown score settings: {unknown}\n"
                    )
                    return 0
                missing = {
                    name: sorted(setting_metric_keys[name] - set(metrics))
                    for name in covered_settings
                    if not setting_metric_keys[name].issubset(metrics)
                }
                if missing:
                    reward_out.write_text("0\n")
                    (out_dir / "score_error.txt").write_text(
                        f"{label} seed {seed} is missing declared-setting metrics: "
                        f"{missing}\n"
                    )
                    return 0

            allowed_score_metrics = set().union(
                *(setting_metric_keys[name] for name in covered_settings)
            )
            cross_setting_metrics = (
                set(metrics) & all_score_metrics
            ) - allowed_score_metrics
            if cross_setting_metrics:
                reward_out.write_text("0\n")
                (out_dir / "score_error.txt").write_text(
                    f"{label} seed {seed} emitted metrics for another setting: "
                    f"{sorted(cross_setting_metrics)}\n"
                )
                return 0

            seen = seen_seed_settings.setdefault(seed, set())
            duplicate = seen.intersection(covered_settings)
            if duplicate:
                reward_out.write_text("0\n")
                (out_dir / "score_error.txt").write_text(
                    f"seed {seed} received duplicate score settings: {sorted(duplicate)}\n"
                )
                return 0
            seen.update(covered_settings)
            seed_metrics = per_seed_metrics.setdefault(seed, {})
            for metric_name in allowed_score_metrics:
                seed_metrics[metric_name] = metrics[metric_name]
            if "elapsed" in log_info:
                try:
                    seed_metrics[f"elapsed_{label}"] = float(log_info["elapsed"])
                except (TypeError, ValueError):
                    pass

    for seed in _config_seeds(config):
        metrics = per_seed_metrics.get(int(seed), {})
        missing_settings = set(setting_metric_keys) - seen_seed_settings.get(int(seed), set())
        if missing_settings:
            reward_out.write_text("0\n")
            (out_dir / "score_error.txt").write_text(
                f"seed {seed} is missing score settings: {sorted(missing_settings)}\n"
            )
            return 0
        _seed_score, _seed_settings, seed_valid = score_record_details(
            spec,
            metrics,
            anchors,
        )
        if not seed_valid:
            reward_out.write_text("0\n")
            (out_dir / "score_error.txt").write_text(
                f"seed {seed} has incomplete, non-finite, or crash-defaulted metrics; "
                "reward forced to 0\n"
            )
            return 0

    valid_metrics = _valid_seed_metric_records(per_seed_metrics)
    if not valid_metrics:
        reward_out.write_text("0\n")
        (out_dir / "score_error.txt").write_text("no metrics extracted from logs\n")
        return 0

    mean_metrics = _aggregate_metrics(valid_metrics)

    # Score via mlsbench DSL against task's score_spec.py + anchors.
    try:
        combined = float(score_record(spec, mean_metrics, anchors))
    except Exception as exc:
        reward_out.write_text("0\n")
        (out_dir / "score_error.txt").write_text(
            f"score computation failed: {type(exc).__name__}: {exc}\n"
        )
        return 0
    if not math.isfinite(combined):
        reward_out.write_text("0\n")
        (out_dir / "score_error.txt").write_text(
            "score computation returned a non-finite value; reward forced to 0\n"
        )
        return 0

    # combined_score is meant to be roughly in [0, 1]; clip defensively.
    reward = max(0.0, min(1.0, combined))

    metrics_text = _canonical_json_text({
        "combined_score": combined,
        "reward": reward,
        "mean_metrics": mean_metrics,
        "per_seed_metrics": per_seed_metrics,
    })
    proof_text = _canonical_json_text({
        "schema_version": 1,
        "status": "passed",
        "strict_fail_closed": True,
        "required_labels": [
            str(tc.get("label", tc.get("cmd", "test")))
            for tc in config.get("test_cmds", [])
        ],
        "required_seeds": _config_seeds(config),
        "reward": reward,
        "metrics_sha256": hashlib.sha256(metrics_text.encode()).hexdigest(),
    })
    _atomic_write_text(out_dir / "metrics.json", metrics_text)
    _atomic_write_text(out_dir / "verification_result.json", proof_text)
    # Publish reward last. If scoring is interrupted before this replace, the
    # zero pre-written by test.sh remains authoritative.
    _atomic_write_text(reward_out, f"{reward}\n")
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
    r.add_argument(
        "--eval-task-meta",
        default=None,
        help="Sanitized runtime task metadata exposed to untrusted eval code.",
    )
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
