#!/usr/bin/env python3
"""Build the per-package Harbor base image.

Produces ``bohanlyu2022/mlsbench-harbor-<pkg>:latest`` from the existing
``bohanlyu2022/mlsbench-<pkg>:latest`` by baking in:

  - the package source at the package's container workdir (so the agent's
    workspace is preloaded with the research scaffold)
  - declared data deps at their      <container_path> (from pkg config)
  - sentinel file                    /opt/mlsbench/workdir (workdir string)

The verifier's pristine diff baseline is shipped per task under
``tests/meta/pristine/`` (mounted at verify time only) rather than baked
into the image, because a root agent could otherwise tamper with an
image-baked baseline before verification.

The harbor adapter's per-task ``task.toml`` then sets
``[environment].docker_image = "bohanlyu2022/mlsbench-harbor-<pkg>:latest"``
so no per-task Docker build is required. Originals are not modified.

Usage:
    python harbor_adapter/scripts/build_base_image.py --package causal-learn
    python harbor_adapter/scripts/build_base_image.py --package causal-learn --push
    python harbor_adapter/scripts/build_base_image.py --all          # every package referenced by any task
    python harbor_adapter/scripts/build_base_image.py --all --push

Pre-requisites:
    - The base image bohanlyu2022/mlsbench-<pkg>:latest must already exist
      locally (pull it with `docker pull`) or be reachable from the daemon.
    - The vendored package source must be present in vendor/<pkg>/ (fetch
      with `mlsbench fetch <pkg>`).
    - Any data_deps with prepare scripts must already be staged on the host
      at vendor/data/<...> (run the prepare scripts manually first).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "harbor_adapter" / "src"))

from mls_bench.adapter import (
    detect_mls_bench_root,
    PREBUILT_DOCKER,
    HARBOR_BASE_DOCKER,
    _load_ops_file,
    _safe_rel_path,
    _safe_join,
    _assert_python_syntax,
    _content_lines,
    _delete_bounds,
    _end_index,
    _normalize_pkg_name,
)


def _apply_pre_edit_ops(mb, pkg: str, root: Path) -> None:
    """Apply only the *package-level* pre_edit.py (which patches the upstream
    source). Per-task mid_edit is applied later, as a tiny per-task Dockerfile
    layer on top of the harbor base image — see adapter.py's task-template
    Dockerfile.j2 and `_stage_task_scaffold`.
    """
    pre_py = mb.pkg_configs_dir / pkg / "pre_edit.py"
    if not pre_py.exists():
        return
    ops = _load_ops_file(pre_py)
    for op in ops:
        target_rel = op.get("file") or ""
        if not target_rel:
            continue
        try:
            parts = _safe_rel_path(target_rel, field="pre_edit file").parts
        except ValueError as exc:
            print(f"  [warn] skipping unsafe pre_edit path {target_rel!r}: {exc}", file=sys.stderr)
            continue
        rel = Path(*parts[1:]) if parts and parts[0] == pkg else Path(target_rel)
        try:
            dst = _safe_join(root, rel.as_posix(), field="pre_edit file")
        except ValueError as exc:
            print(f"  [warn] skipping unsafe pre_edit destination {target_rel!r}: {exc}", file=sys.stderr)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        kind = op.get("op")
        content = op.get("content", "")
        if kind == "create":
            new_text = "".join(_content_lines(content))
        elif kind == "replace" and dst.exists():
            lines = dst.read_text().splitlines(keepends=True)
            s = int(op["start_line"]) - 1
            e = _end_index(lines, int(op["end_line"]))
            new_text = "".join(lines[:s] + _content_lines(content) + lines[e:])
        elif kind == "insert":
            existing = dst.read_text() if dst.exists() else ""
            lines = existing.splitlines(keepends=True)
            after = int(op.get("after_line", 0))
            new_text = "".join(lines[:after] + _content_lines(content) + lines[after:])
        elif kind == "delete" and dst.exists():
            lines = dst.read_text().splitlines(keepends=True)
            start, end = _delete_bounds(op)
            s = start - 1
            e = _end_index(lines, end)
            new_text = "".join(lines[:s]) + "".join(lines[e:])
        else:
            continue
        _assert_python_syntax(
            new_text,
            rel_path=str(rel.as_posix()),
            origin=f"pre_edit.py for {pkg}",
            task_id=f"base-image:{pkg}",
        )
        dst.write_text(new_text)


def _render_dockerfile(
    workdir: str,
    pkg: str,
    data_entries: list[tuple[str, str]],
    pkg_env: dict[str, str] | None = None,
) -> str:
    """Dockerfile that reads inputs via BuildKit named build contexts so we
    skip the /tmp staging copy entirely. ``data_entries`` is
    [(context_name, container_path), …] for each data dep. ``pkg_env`` is
    the pkg_config ``env`` dict — baked as ENV so things like ``HF_HOME``
    take effect for the agent's interactive shell too (the verifier already
    re-injects these via package_envs.json, but the agent doesn't get that
    pass).
    """
    pkg_env = pkg_env or {}
    lines = [
        "# syntax=docker/dockerfile:1.6",
        "ARG BASE_IMAGE",
        "FROM ${BASE_IMAGE}",
        "",
        "ENV DEBIAN_FRONTEND=noninteractive \\",
        "    PYTHONUNBUFFERED=1",
        "",
        "# Live package source at the agent's workdir.",
        # NOTE: The pristine reference for the verifier's edit-range diff
        # guard is NOT baked here. Agent runs as root and could tamper with
        # any image-baked pristine before verify. Instead the adapter ships
        # pristine assets under tests/meta/pristine/, which Harbor mounts at
        # /tests/ only at verify time. See adapter.py::_stage_verifier_assets
        # and score_task.py::cmd_guard.
        f"COPY --from=workspace . {workdir}/{pkg}/",
        "",
        # NOTE: We intentionally do NOT bake mlsbench source into the image.
        # parser.py / score_spec.py / score_task.py only need mlsbench at
        # verify time, which is precisely when Harbor mounts the per-task
        # tests/ directory at /tests/. The adapter writes a copy of
        # mlsbench src into tests/mlsbench_src/ (see
        # adapter.py::_stage_verifier_assets) and score_task.py adds
        # /tests/mlsbench_src to sys.path. Agent shell never sees it.
        "",
    ]
    if data_entries:
        lines.append("# Data deps from pkg_config.")
        for name, container in data_entries:
            lines.append(f"COPY --from={name} . {container}/")
    else:
        lines.append("# (no data deps)")
    if pkg_env:
        lines.extend(["", "# pkg_config.env, baked so the agent shell sees them too."])
        for k, v in pkg_env.items():
            # ENV expects a single line; escape internal newlines defensively.
            v_str = str(v).replace("\n", " ")
            lines.append(f"ENV {k}={v_str}")
    lines.extend([
        "",
        "# Sentinel + ENTRYPOINT [] so Harbor's `sh -c sleep infinity` runs as-is",
        "# (base image has ENTRYPOINT=[bash] which would otherwise break it).",
        "COPY workdir /opt/mlsbench/workdir",
        "ENTRYPOINT []",
        f"WORKDIR {workdir}",
        'CMD ["/bin/bash"]',
        "",
    ])
    return "\n".join(lines)


def _du_hardlink_aware(p: Path) -> int:
    if not p.exists(): return 0
    seen: set[tuple[int,int]] = set()
    total = 0
    for f in p.rglob("*"):
        if not f.is_file(): continue
        try:
            st = f.stat()
        except OSError:
            continue
        key = (st.st_dev, st.st_ino)
        if key in seen: continue
        seen.add(key)
        total += st.st_size
    return total


def _resolve_data_deps(
    mb,
    pkg_config: dict,
    *,
    max_total_gb: float | None = None,
) -> tuple[list[tuple[Path, str, str]], list[str]]:
    """Return ([(host_src, container_path, dep_name), …], warnings).

    Filters out:
      - missing host paths
      - whole-data_root host paths (a known pkg_config misconfig: TSL declares
        host_path = '{data_root}' which means "bind mount everything", harmless
        for MLS-Bench native runtime but disastrous for image baking).

    Raises a ValueError if the total resolved data exceeds ``max_total_gb``;
    callers should treat this as a skip-this-package signal.
    """
    out: list[tuple[Path, str, str]] = []
    warnings: list[str] = []
    data_root = mb.vendor_dir / "data"
    data_root_resolved = data_root.resolve()
    project_root = mb.root
    total_bytes = 0
    for dep in pkg_config.get("data_deps") or []:
        name = dep.get("name", "")
        host_template = (
            str(dep.get("host_path", ""))
            .replace("{project_root}", str(project_root))
            .replace("{data_root}", str(data_root))
        )
        host = Path(host_template).expanduser()
        container = str(dep.get("container_path", "")).rstrip("/")
        if not container:
            continue
        if not host.exists():
            warnings.append(
                f"data dep '{name}' missing on host at {host}; "
                "skipping (eval scripts will fail if they need this data)."
            )
            continue
        # Reject the "host_path = data_root" misconfig.
        try:
            same_as_root = host.resolve() == data_root_resolved
        except OSError:
            same_as_root = False
        if same_as_root:
            # MLS-Bench convention for packages whose base image's install_cmds
            # populates /data directly (e.g. Time-Series-Library does
            # snapshot_download('thuml/Time-Series-Library', local_dir='/data')
            # at base-image-build time). host_path = data_root just tells the
            # native runtime to bind-mount the whole vendor/data tree at /data.
            # For our harbor base build we don't COPY anything — the data is
            # already baked into the base image via its install_cmds, and we
            # FROM that base. No size penalty, no /tmp staging.
            warnings.append(
                f"data dep '{name}': host_path == data_root → inheriting from "
                "base image (install_cmds bakes /data)."
            )
            continue
        sz = _du_hardlink_aware(host)
        total_bytes += sz
        out.append((host, container, name))
    if max_total_gb is not None and total_bytes > max_total_gb * 1e9:
        raise ValueError(
            f"package data deps total {total_bytes/1e9:.1f} GB > "
            f"--max-data-gb {max_total_gb:.0f}; skipping"
        )
    return out, warnings


def _resolve_data_binds(
    mb,
    pkg_config: dict,
) -> tuple[list[tuple[Path, str, str]], list[str]]:
    """Resolve ``pkg_config['data_bind']`` ("src:target") into bake tuples.

    Harbor has NO runtime bind-mount support, so every native ``data_bind``
    must be baked into the image at its target path (otherwise files the eval
    scripts read from e.g. ``/workspace/<pkg>/cache`` simply aren't there).
    Mirrors ``_resolve_data_deps``' {project_root}/{data_root} substitution
    and missing-host skipping. Returns ([(host, container, name), …], warnings).
    """
    out: list[tuple[Path, str, str]] = []
    warnings: list[str] = []
    data_root = mb.vendor_dir / "data"
    data_root_resolved = data_root.resolve()
    project_root = mb.root
    # data_bind may be a single "src:target" string or a list of them.
    raw_binds = pkg_config.get("data_bind") or []
    if isinstance(raw_binds, str):
        raw_binds = [raw_binds]
    for i, raw in enumerate(raw_binds):
        spec = str(raw)
        if ":" not in spec:
            warnings.append(f"data_bind entry {spec!r} has no ':target'; skipping")
            continue
        src_s, target = spec.rsplit(":", 1)
        src_s = (
            src_s.replace("{project_root}", str(project_root))
            .replace("{data_root}", str(data_root))
        )
        host = Path(src_s).expanduser()
        container = target.rstrip("/")
        if not container:
            continue
        if not host.exists():
            warnings.append(
                f"data_bind src missing on host at {host}; skipping "
                "(Harbor eval will fail if it needs this data)."
            )
            continue
        # Same guard as _resolve_data_deps: never bake the whole shared
        # vendor/data tree (some packages bind "{data_root}:/data").
        try:
            if host.resolve() == data_root_resolved:
                warnings.append(
                    f"data_bind src == data_root ({host}); skipping to avoid "
                    "baking the entire shared data tree into the image."
                )
                continue
        except OSError:
            pass
        out.append((host, container, f"bind_{i}"))
    return out, warnings


def _dedupe_bake_entries(
    entries: list[tuple[Path, str, str]],
) -> list[tuple[Path, str, str]]:
    """Drop exact (resolved-host, container) duplicates only.

    Intentionally does NOT drop ancestor-host entries: a package may map a
    parent dir (with files directly under it) to one container AND a subdir to
    another, and dropping the parent would lose the parent's own files (e.g.
    dbim-codebase). The cost is that a package mapping the same bytes to two
    containers bakes them twice — correct, just larger; preferred over wrong.
    """
    seen: set[tuple[str, str]] = set()
    uniq: list[tuple[Path, str, str]] = []
    for host, container, name in entries:
        try:
            hr = str(host.resolve())
        except OSError:
            hr = str(host)
        key = (hr, container)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((host, container, name))
    return uniq


def build_one(
    pkg: str,
    *,
    push: bool = False,
    mb_root: Path | None = None,
    max_data_gb: float | None = None,
    skip_if_pushed: bool = False,
    push_executor=None,
    push_futures: list | None = None,
) -> int:
    """If ``push_executor`` is set, do the build synchronously and submit the
    push to the executor (non-blocking); the caller is responsible for
    eventually collecting ``push_futures``. Without an executor, build and
    push happen sequentially in the same process.
    """
    mb = detect_mls_bench_root(mb_root)
    pkg_src = mb.package_src(pkg)
    if pkg_src is None:
        print(
            f"[error] package source for '{pkg}' not found at "
            f"vendor/external_packages/{pkg}/ or vendor/{pkg}/.",
            file=sys.stderr,
        )
        return 2

    pkg_cfg_path = mb.pkg_configs_dir / pkg / "config.json"
    if not pkg_cfg_path.exists():
        print(f"[error] vendor/pkg_configs/{pkg}/config.json missing", file=sys.stderr)
        return 2
    pkg_cfg = json.loads(pkg_cfg_path.read_text())
    workdir = pkg_cfg.get("workdir", "/workspace")

    base = PREBUILT_DOCKER.format(pkg=pkg.lower())
    target = HARBOR_BASE_DOCKER.format(pkg=pkg.lower())

    if skip_if_pushed:
        rc = subprocess.run(
            ["docker", "manifest", "inspect", target],
            capture_output=True,
        ).returncode
        if rc == 0:
            print(f"[{pkg}] already on Docker Hub at {target}; skipping", file=sys.stderr)
            return 0

    try:
        data_entries, data_warnings = _resolve_data_deps(
            mb, pkg_cfg, max_total_gb=max_data_gb,
        )
    except ValueError as exc:
        print(f"[skip] {pkg}: {exc}", file=sys.stderr)
        return 3
    for w in data_warnings:
        print(f"[{pkg}] [warn] {w}", file=sys.stderr)

    # Harbor can't bind-mount, so bake data_bind targets into the image too.
    # Dedupe so shared bytes (e.g. a big cache/ subdir) aren't baked twice.
    bind_entries, bind_warnings = _resolve_data_binds(mb, pkg_cfg)
    for w in bind_warnings:
        print(f"[{pkg}] [warn] {w}", file=sys.stderr)
    data_entries = _dedupe_bake_entries(list(data_entries) + bind_entries)
    # Re-apply the size cap to the COMBINED set (binds bypass _resolve_data_deps' check).
    if max_data_gb is not None:
        combined_gb = sum(_du_hardlink_aware(h) for h, _, _ in data_entries) / 1e9
        if combined_gb > max_data_gb:
            print(
                f"[skip] {pkg}: baked data {combined_gb:.1f} GB > "
                f"--max-data-gb {max_data_gb:.0f}",
                file=sys.stderr,
            )
            return 3

    with tempfile.TemporaryDirectory(prefix=f"mlsbench-harbor-{pkg}-") as td:
        ctx = Path(td)

        # If pre_edit ops exist we need a writable copy to mutate; otherwise
        # we can reference the package source directly as a build context.
        workspace_ctx = pkg_src
        pre_py = mb.pkg_configs_dir / pkg / "pre_edit.py"
        if pre_py.exists():
            mat = ctx / "_pristine_with_pre_edit" / pkg
            try:
                shutil.copytree(
                    pkg_src, mat,
                    ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
                    symlinks=False,
                    ignore_dangling_symlinks=True,
                )
            except shutil.Error as exc:
                # Some vendored packages contain broken symlinks (e.g.
                # SMPyBandits/docs/paper/plots/paper -> nonexistent). Log and
                # continue; the dangling links don't affect runtime.
                print(
                    f"[{pkg}] [warn] shutil.copytree encountered {len(exc.args[0])} "
                    f"non-fatal error(s); continuing",
                    file=sys.stderr,
                )
            _apply_pre_edit_ops(mb, pkg, mat)
            workspace_ctx = mat

        (ctx / "workdir").write_text(workdir + "\n")

        # BuildKit named build contexts. Each name maps to a host directory
        # the docker daemon reads directly — no /tmp staging copy.
        # `pristine` context dropped: pristine reference is shipped under
        # tests/meta/pristine/ at adapter render time, not baked into the
        # image. See _render_dockerfile.
        build_contexts: list[tuple[str, Path]] = [
            ("workspace", workspace_ctx),
            # NOTE: mlsbench src is NOT baked into the image — it ships in
            # the per-task tests/ dir and Harbor mounts it only at verify
            # time. See adapter.py::_stage_verifier_assets.
        ]
        data_ctx_entries: list[tuple[str, str]] = []
        for i, (host, container, name) in enumerate(data_entries):
            ctx_name = f"data_{i}"
            build_contexts.append((ctx_name, host))
            data_ctx_entries.append((ctx_name, container))

        (ctx / "Dockerfile").write_text(_render_dockerfile(
            workdir=workdir, pkg=pkg, data_entries=data_ctx_entries,
            pkg_env=pkg_cfg.get("env") or {},
        ))

        argv = [
            "docker", "buildx", "build",
            "--build-arg", f"BASE_IMAGE={base}",
            "-t", target, "-f", str(ctx / "Dockerfile"),
            "--load",          # load into local docker daemon
            "--pull=false",    # never pull base image; the local copy is canonical
        ]
        for name, path in build_contexts:
            argv += ["--build-context", f"{name}={path}"]
        argv.append(str(ctx))

        total_data_gb = sum(_du_hardlink_aware(h) for h, _, _ in data_entries) / 1e9
        print(
            f"[{pkg}] building {target} FROM {base} "
            f"({len(data_entries)} data dep(s), ~{total_data_gb:.1f} GB)",
            file=sys.stderr,
        )
        rc = subprocess.run(argv).returncode
        if rc != 0:
            print(f"[error] docker buildx build failed for {pkg}", file=sys.stderr)
            return rc

        if push:
            if push_executor is not None and push_futures is not None:
                fut = push_executor.submit(_push_image, pkg, target)
                push_futures.append(fut)
                print(f"[{pkg}] build done; push queued in background", file=sys.stderr)
            else:
                print(f"[{pkg}] pushing {target}", file=sys.stderr)
                rc = subprocess.run(["docker", "push", target]).returncode
                if rc != 0:
                    print(f"[error] docker push failed for {pkg}", file=sys.stderr)
                    return rc
                print(f"[{pkg}] done -> {target}", file=sys.stderr)
        else:
            print(f"[{pkg}] done -> {target}", file=sys.stderr)
    return 0


def _push_image(pkg: str, target: str) -> tuple[str, int]:
    """Run ``docker push`` for a single target. Returns (pkg, returncode)."""
    print(f"[{pkg}] pushing {target}", file=sys.stderr, flush=True)
    rc = subprocess.run(["docker", "push", target]).returncode
    if rc != 0:
        print(f"[error] docker push failed for {pkg}", file=sys.stderr, flush=True)
    else:
        print(f"[{pkg}] pushed -> {target}", file=sys.stderr, flush=True)
    return pkg, rc


def _all_packages_referenced_by_tasks(mb) -> list[str]:
    pkgs: dict[str, str] = {}
    for t in mb.list_tasks():
        cfg = json.loads((mb.tasks_dir / t / "config.json").read_text())
        for tc in cfg.get("test_cmds", []):
            if tc.get("package"):
                pkg = str(tc["package"])
                pkgs.setdefault(_normalize_pkg_name(pkg), pkg)
    return sorted(pkgs.values(), key=str.lower)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--package", help="One package name, e.g. causal-learn")
    g.add_argument("--all", action="store_true",
                   help="Build for every package referenced by any task")
    p.add_argument("--push", action="store_true",
                   help="Push the resulting image to the configured registry")
    p.add_argument("--mls-bench-root", type=Path, default=None)
    p.add_argument("--max-data-gb", type=float, default=None,
                   help="Skip any package whose to-be-COPYed data deps total "
                        "more than N GB (host_path=data_root deps are NOT "
                        "counted; they inherit from the base image). Useful "
                        "to avoid pushing impossibly large images.")
    p.add_argument("--skip-if-pushed", action="store_true",
                   help="Skip packages whose harbor base image already exists "
                        "on the registry (resume-friendly).")
    p.add_argument("--skip-pkg", action="append", default=[],
                   help="Skip a package entirely (neither built nor pushed). "
                        "Repeat for multiple packages. Matched case- and "
                        "separator-insensitively against `vendor/pkg_configs/<pkg>`.")
    p.add_argument("--push-workers", type=int, default=2,
                   help="Number of concurrent `docker push` workers. The main "
                        "loop keeps building locally while pushes run in the "
                        "background. Set to 0 to disable parallelism (build "
                        "and push strictly serial).")
    args = p.parse_args(argv)

    mb = detect_mls_bench_root(args.mls_bench_root)

    if args.all:
        pkgs = _all_packages_referenced_by_tasks(mb)
        print(f"Building harbor base images for {len(pkgs)} packages", file=sys.stderr)
    else:
        pkgs = [args.package]

    if args.skip_pkg:
        skip_norm = {_normalize_pkg_name(s) for s in args.skip_pkg}
        before = len(pkgs)
        pkgs = [p for p in pkgs if _normalize_pkg_name(p) not in skip_norm]
        skipped = before - len(pkgs)
        if skipped:
            print(
                f"--skip-pkg removed {skipped} package(s): {sorted(args.skip_pkg)}",
                file=sys.stderr,
            )

    summary = {"ok": [], "failed": [], "skipped": []}
    push_executor = None
    push_futures: list = []
    if args.push and args.push_workers > 0:
        from concurrent.futures import ThreadPoolExecutor
        push_executor = ThreadPoolExecutor(
            max_workers=args.push_workers, thread_name_prefix="push"
        )
        print(
            f"Push workers: {args.push_workers} (build + push run in parallel)",
            file=sys.stderr,
        )

    try:
        for pkg in pkgs:
            try:
                r = build_one(
                    pkg,
                    push=args.push,
                    mb_root=args.mls_bench_root,
                    max_data_gb=args.max_data_gb,
                    skip_if_pushed=args.skip_if_pushed,
                    push_executor=push_executor,
                    push_futures=push_futures,
                )
            except Exception as exc:
                import traceback
                traceback.print_exception(type(exc), exc, exc.__traceback__)
                print(
                    f"[{pkg}] [error] build_one raised {type(exc).__name__}: {exc}; "
                    "marking failed and continuing to next package",
                    file=sys.stderr,
                )
                summary["failed"].append(pkg)
                continue
            if r == 0:
                summary["ok"].append(pkg)
            elif r == 3:
                summary["skipped"].append(pkg)
            else:
                summary["failed"].append(pkg)
    finally:
        if push_executor is not None:
            print(
                f"Waiting on {len(push_futures)} pending pushes to finish ...",
                file=sys.stderr,
            )
            for fut in push_futures:
                try:
                    pkg_name, rc = fut.result()
                except Exception as exc:
                    # Catching here is essential — a thread-side exception
                    # (network error, API timeout, killed by SIGINT) would
                    # otherwise propagate and silently leave the package in
                    # the OK tally. We can't recover the pkg name from the
                    # raised exception, so log + best-effort mark unknown.
                    import traceback
                    traceback.print_exception(type(exc), exc, exc.__traceback__)
                    print(
                        f"[error] push worker raised: {exc}; "
                        "marking as failed (pkg unknown).",
                        file=sys.stderr,
                    )
                    summary["failed"].append("<unknown-push-exception>")
                    continue
                if rc != 0:
                    if pkg_name in summary["ok"]:
                        summary["ok"].remove(pkg_name)
                    if pkg_name not in summary["failed"]:
                        summary["failed"].append(pkg_name)
            push_executor.shutdown(wait=True)

    print(
        f"\n=== summary ===\nok: {len(summary['ok'])}, "
        f"skipped: {len(summary['skipped'])}, "
        f"failed: {len(summary['failed'])}",
        file=sys.stderr,
    )
    if summary["skipped"]:
        print(f"skipped: {summary['skipped']}", file=sys.stderr)
    if summary["failed"]:
        print(f"failed:  {summary['failed']}", file=sys.stderr)
    return 0 if not summary["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
