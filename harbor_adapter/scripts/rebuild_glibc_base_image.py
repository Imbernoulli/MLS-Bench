#!/usr/bin/env python3
"""Rebuild a per-package Harbor base ``.sif`` on a newer-glibc host OS.

Root cause (confirmed by hand for ``opt-vr-bench`` and ``pytorch-geometric``
on 2026-08-19 -- see conversation/issue history for the full writeup):
Harbor's stock ``SingularityEnvironment._start_server()`` always launches the
container via ``singularity exec --fakeroot``. On a cluster with no
``/etc/subuid`` entry for the invoking user (true here), apptainer can't do a
real kernel-level rootless user-namespace, so it falls back to a userspace
``faked`` helper binary that's dynamically linked against the *host's*
glibc. Every ``bohanlyu2022/mlsbench-harbor-<pkg>:latest`` image built from a
``pytorch/pytorch:*`` tag ships Ubuntu 22.04 (glibc 2.35) -- across every
PyTorch version checked, 2.1.2 through 2.6.0 -- which is too old to satisfy
that host-linked ``faked`` binary on this cluster (Rocky Linux 10.1, glibc
2.39). Result: ``fakeroot: error while starting the 'faked' daemon`` /
``GLIBC_2.38 not found``, and the trial never starts. ~49 of the 52 packages
used by GPU tasks are on a ``pytorch/pytorch:*`` base and are almost
certainly affected; ``cleanrl``/``CORL`` (``nvidia/cuda:*-ubuntu20.04``,
glibc 2.31) are even older.

Fix: extract the package source + data deps + the ``/opt/mlsbench/workdir``
sentinel from the existing (broken) base ``.sif``, and re-layer them onto a
fresh ``ubuntu:24.04`` (glibc 2.39) image with the same Python packages
reinstalled via pip. No cluster-admin / subuid change needed. This is a
narrower, more surgical fix than getting subuid ranges provisioned
cluster-wide (which would fix every image, current and future, without a
per-package rebuild) -- see the discussion this script came out of before
deciding which one to invest in further.

This script does NOT touch anything under ``bohanlyu2022/`` on Docker Hub or
the shared ``harbor_singularity_cache`` -- it only ever writes into
``--output-cache-dir`` (default: a sibling ``_glibc239`` directory). Point
``SingularityGPUEnvironment``'s ``singularity_image_cache_dir`` kwarg at that
directory (in a Harbor job config) to actually use a rebuilt image for a run.

Two base-image families are auto-handled:

  * ``pytorch/pytorch:<torch>-cuda<X.Y>-cudnn<N>-{devel,runtime}`` -- the
    base image supplies torch/cuda; the package's own ``install_cmds``
    usually does NOT reinstall torch, so this script prepends an explicit
    ``pip install torch==<torch> --index-url .../whl/cu<XY>`` step before
    replaying the package's ``install_cmds`` verbatim.
  * ``nvidia/cuda:<ver>-...-ubuntu<rel>`` -- these ship no Python at all;
    every example checked has ``install_cmds`` that already do their own
    ``apt-get install python3-pip`` + (if needed) an explicit
    ``pip install torch==... --index-url ...`` line, so this script just
    ensures ``ca-certificates`` is present and replays ``install_cmds``
    verbatim with no extra prepend.

Everything else (``continuumio/miniconda3:*``, ``nvcr.io/nvidia/pytorch:*``
NGC images, custom images like ``verlai/verl:*``, or any package whose
``install_cmds`` calls ``conda``) is classified ``needs-review`` and SKIPPED
by default -- ``--list`` shows why. ``humanoid-gym`` in particular layers in
IsaacGym + Vulkan ICD + a CUDA-forward-compat driver hack that's already
special-cased elsewhere (see harbor_adapter/README.md's "IsaacGym on Hopper"
section) and should not be blindly rebuilt by this script.

Usage (must run on a node with ``apptainer`` on PATH -- a GPU compute node
on this cluster via ``srun --partition=gpu ...``, NOT the login node; a GPU
itself is only needed to *run* the rebuilt image afterwards, not to build
it):

    # Survey scope first -- no build, no node requirements beyond reading configs.
    python rebuild_glibc_base_image.py --list

    # Build (or --dry-run to just print the .def and skip the apptainer build):
    python rebuild_glibc_base_image.py --pkg opt-vr-bench
    python rebuild_glibc_base_image.py --pkg opt-vr-bench --pkg pytorch-geometric
    python rebuild_glibc_base_image.py --all-gpu-packages
    python rebuild_glibc_base_image.py --all-gpu-packages --only-strategy pytorch

    # Then point a Harbor job config at the rebuilt images, e.g.:
    #   environment:
    #     import_path: mls_bench.harbor_env:SingularityGPUEnvironment
    #     kwargs:
    #       singularity_image_cache_dir: /mnt/beegfs/work/fang/harbor_singularity_cache_glibc239
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PKG_CONFIGS_DIR = REPO_ROOT / "vendor" / "pkg_configs"
HARBOR_TASKS_DIR = REPO_ROOT / "harbor" / "tasks"

# glibc floor for the rebuilt image. Bump this (and re-verify against `ldd
# --version` / `apptainer exec --fakeroot <sif> whoami` on a real node) only
# if the cluster's own host OS ever moves off Rocky Linux 10.1 (glibc 2.39).
NEW_BASE_OS = "ubuntu:24.04"

DEFAULT_SOURCE_CACHE_DIR = Path("/mnt/beegfs/work/fang/harbor_singularity_cache")
DEFAULT_OUTPUT_CACHE_DIR = Path("/mnt/beegfs/work/fang/harbor_singularity_cache_glibc239")
DEFAULT_WORK_DIR = Path("/mnt/beegfs/work/fang/claude_scratch/glibc_rebuild_work")

HARBOR_BASE_IMAGE_TEMPLATE = "bohanlyu2022/mlsbench-harbor-{pkg}:latest"

_PYTORCH_TAG_RE = re.compile(
    r"^pytorch/pytorch:(?P<torch>\d+\.\d+\.\d+)-cuda(?P<cuda>\d+\.\d+)-cudnn\d+-(?P<kind>devel|runtime)$"
)
_NVIDIA_CUDA_TAG_RE = re.compile(r"^nvidia/cuda:")


class Strategy:
    PYTORCH = "pytorch"
    NVIDIA_CUDA = "nvidia-cuda"
    NEEDS_REVIEW = "needs-review"


def load_pkg_config(pkg: str) -> dict:
    path = PKG_CONFIGS_DIR / pkg / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"No pkg_config for {pkg!r}: {path}")
    return json.loads(path.read_text())


def discover_gpu_packages() -> list[str]:
    """Every mls_bench_package referenced by a task.toml with gpus > 0."""
    pkgs: set[str] = set()
    for task_toml in sorted(HARBOR_TASKS_DIR.glob("*/task.toml")):
        text = task_toml.read_text()
        m_gpus = re.search(r"^gpus\s*=\s*(\d+)", text, re.MULTILINE)
        if not m_gpus or int(m_gpus.group(1)) == 0:
            continue
        m_pkg = re.search(r'^mls_bench_package\s*=\s*"([^"]+)"', text, re.MULTILINE)
        if m_pkg:
            pkgs.add(m_pkg.group(1))
    return sorted(pkgs)


def classify(config: dict) -> tuple[str, str]:
    """Return (strategy, note)."""
    base_image = config.get("base_image", "")
    install_cmds = config.get("install_cmds", [])
    if any("conda install" in c or "conda create" in c or "conda env" in c for c in install_cmds):
        return Strategy.NEEDS_REVIEW, "install_cmds calls conda; needs manual translation"
    if _PYTORCH_TAG_RE.match(base_image):
        return Strategy.PYTORCH, ""
    if _NVIDIA_CUDA_TAG_RE.match(base_image):
        return Strategy.NVIDIA_CUDA, ""
    return Strategy.NEEDS_REVIEW, f"unrecognized base_image {base_image!r}"


def _parse_pytorch_tag(base_image: str) -> dict:
    m = _PYTORCH_TAG_RE.match(base_image)
    assert m, base_image
    cuda_tag = "cu" + m.group("cuda").replace(".", "")
    return {"torch": m.group("torch"), "cuda_tag": cuda_tag, "devel": m.group("kind") == "devel"}


def extract_paths_for(config: dict) -> list[str]:
    """Container paths to pull out of the existing (broken) base sif.

    Always: the package workdir (source), /data (declared data_deps land
    here by convention in most configs), and /opt/mlsbench (the sentinel
    build_base_image.py bakes in). Plus anything a package's own
    extra_files/data_deps points at outside those (e.g. humanoid-gym's
    /opt/isaacgym) -- though packages needing that are usually
    needs-review anyway.
    """
    workdir = config.get("workdir", "/workspace").rstrip("/")
    paths = {workdir, "/data", "/opt/mlsbench"}
    for ef in config.get("extra_files", []):
        dst = ef.get("dst")
        if dst:
            paths.add(dst.rstrip("/"))
    for dd in config.get("data_deps", []):
        cp = dd.get("container_path")
        if cp:
            paths.add(cp.rstrip("/"))
    return sorted(paths)


def _sanitize(container_path: str) -> str:
    return container_path.strip("/").replace("/", "__") or "root"


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"+ {' '.join(cmd)}", file=sys.stderr)
    return subprocess.run(cmd, check=True, **kwargs)


def ensure_source_sif(pkg: str, source_cache_dir: Path) -> Path:
    """Return the path to the existing (possibly glibc-broken) harbor base
    sif, pulling it fresh via apptainer if not already cached.

    Uses the same filename convention as
    harbor.environments.singularity.singularity.SingularityEnvironment._convert_docker_to_sif
    (docker_image.replace('/', '_').replace(':', '_') + '.sif') so this can
    reuse whatever the real harbor_singularity_cache already has pulled.
    """
    docker_image = HARBOR_BASE_IMAGE_TEMPLATE.format(pkg=pkg)
    safe_name = docker_image.replace("/", "_").replace(":", "_")
    sif_path = source_cache_dir / f"{safe_name}.sif"
    if sif_path.exists():
        return sif_path
    source_cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{pkg}] pulling {docker_image} -> {sif_path} (not cached locally)", file=sys.stderr)
    run(["apptainer", "pull", str(sif_path), f"docker://{docker_image}"])
    return sif_path


def extract_from_sif(pkg: str, source_sif: Path, container_paths: list[str], work_dir: Path) -> dict[str, Path]:
    """Extract each container_path from source_sif to work_dir/extracted/<sanitized>/.

    No --fakeroot: this is the exact sif that can't run its own fakeroot
    helper, and plain read-only `tar -c` needs no privilege bump anyway.
    Returns {container_path: host_dir}.
    """
    extract_root = work_dir / pkg / "extracted"
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True)

    out: dict[str, Path] = {}
    for cpath in container_paths:
        host_dir = extract_root / _sanitize(cpath)
        host_dir.mkdir(parents=True, exist_ok=True)
        tar_proc = subprocess.Popen(
            ["apptainer", "exec", "--containall", str(source_sif), "bash", "-c",
             f"tar -C {cpath} -cf - . 2>/dev/null || true"],
            stdout=subprocess.PIPE,
        )
        extract_proc = subprocess.run(
            ["tar", "-C", str(host_dir), "-xf", "-"],
            stdin=tar_proc.stdout,
        )
        tar_proc.stdout.close()
        tar_proc.wait()
        if extract_proc.returncode != 0:
            print(f"[{pkg}] warning: extracting {cpath} exited nonzero (may not exist in image)", file=sys.stderr)
        out[cpath] = host_dir
    return out


def render_def_file(
    pkg: str,
    config: dict,
    strategy: str,
    extracted: dict[str, Path],
) -> str:
    lines = [
        f"Bootstrap: docker",
        f"From: {NEW_BASE_OS}",
        "",
        "%setup",
        "    # This cluster's global apptainer config bind-mounts /storage into",
        "    # every container at start (even during %post); create the mount",
        "    # point on the host side before that's attempted.",
        "    mkdir -p ${APPTAINER_ROOTFS}/storage",
        "",
        "%files",
    ]
    for cpath, host_dir in extracted.items():
        lines.append(f"    {host_dir} {cpath}")
    lines += [
        "",
        "%post",
        "    export DEBIAN_FRONTEND=noninteractive",
        # NEW_BASE_OS (ubuntu:24.04) enforces PEP 668: a bare `pip install`
        # fails with "externally-managed-environment". The original bases
        # (pytorch/pytorch's conda env, nvidia/cuda:*-ubuntu20.04's system
        # pip) didn't enforce this, so every package's install_cmds was
        # written assuming unrestricted pip. Set this instead of rewriting
        # every config's install_cmds.
        "    export PIP_BREAK_SYSTEM_PACKAGES=1",
    ]

    if strategy == Strategy.PYTORCH:
        info = _parse_pytorch_tag(config["base_image"])
        lines += [
            "    apt-get update -q",
            "    apt-get install -y --no-install-recommends python3 python3-pip python-is-python3 git wget build-essential ca-certificates",
            "    rm -rf /var/lib/apt/lists/*",
            f"    pip install --no-cache-dir --break-system-packages torch=={info['torch']} --index-url https://download.pytorch.org/whl/{info['cuda_tag']}",
        ]
    elif strategy == Strategy.NVIDIA_CUDA:
        # No python3 preamble here: unlike the pytorch/pytorch base (which
        # already has a `python` on PATH via conda, so its install_cmds never
        # set one up), nvidia/cuda:* base images ship no Python at all, and
        # this strategy's install_cmds always install python3 themselves and
        # symlink `python` explicitly (verified for cleanrl and CORL).
        lines += [
            "    apt-get update -q",
            "    apt-get install -y --no-install-recommends ca-certificates",
            "    rm -rf /var/lib/apt/lists/*",
        ]
    else:
        raise ValueError(f"render_def_file called with unhandled strategy {strategy!r} for {pkg}")

    for cmd in config.get("install_cmds", []):
        # %post is already a literal bash script body (unlike a Dockerfile
        # RUN instruction), so a multi-line entry -- e.g. a `python -c "..."`
        # call with an embedded multi-line string -- can be pasted in
        # verbatim with no heredoc wrapping and no added indentation: both
        # would corrupt the embedded string's literal newlines/whitespace
        # (Python is indentation-sensitive) or, for the heredoc terminator
        # specifically, break bash's requirement that an unquoted `<<'X'`
        # closing delimiter start at column 0.
        lines.append(cmd)

    env = config.get("env", {})
    if env:
        lines += ["", "%environment"]
        for k, v in env.items():
            lines.append(f"    export {k}={v}")

    return "\n".join(lines) + "\n"


def build_one(
    pkg: str,
    *,
    source_cache_dir: Path,
    output_cache_dir: Path,
    work_dir: Path,
    force: bool,
    dry_run: bool,
    allow_needs_review: bool,
) -> None:
    config = load_pkg_config(pkg)
    strategy, note = classify(config)
    print(f"[{pkg}] base_image={config.get('base_image')!r} strategy={strategy}" + (f" ({note})" if note else ""))

    if strategy == Strategy.NEEDS_REVIEW and not allow_needs_review:
        print(f"[{pkg}] SKIPPING: needs-review ({note}). Pass --allow-needs-review to attempt anyway (best-effort, unverified).")
        return

    output_cache_dir.mkdir(parents=True, exist_ok=True)
    docker_image = HARBOR_BASE_IMAGE_TEMPLATE.format(pkg=pkg)
    safe_name = docker_image.replace("/", "_").replace(":", "_")
    out_sif = output_cache_dir / f"{safe_name}.sif"
    if out_sif.exists() and not force:
        print(f"[{pkg}] already built at {out_sif} (pass --force to rebuild)")
        return

    source_sif = ensure_source_sif(pkg, source_cache_dir)
    container_paths = extract_paths_for(config)
    print(f"[{pkg}] extracting {container_paths} from {source_sif}")
    extracted = extract_from_sif(pkg, source_sif, container_paths, work_dir)

    def_text = render_def_file(pkg, config, strategy, extracted)
    def_path = work_dir / pkg / f"{pkg}-glibc239.def"
    def_path.write_text(def_text)
    print(f"[{pkg}] wrote {def_path}")

    if dry_run:
        print(def_text)
        print(f"[{pkg}] --dry-run: skipping apptainer build")
        return

    print(f"[{pkg}] building {out_sif} (this can take several minutes)")
    run(["apptainer", "build", "--fakeroot", "--force", str(out_sif), str(def_path)])

    print(f"[{pkg}] sanity check: fakeroot exec on rebuilt sif")
    run([
        "apptainer", "exec", "--fakeroot", "--containall", str(out_sif),
        "bash", "-c", "ldd --version | head -1",
    ])
    print(f"[{pkg}] done: {out_sif}")


def cmd_list(pkgs: list[str]) -> None:
    for pkg in pkgs:
        try:
            config = load_pkg_config(pkg)
        except FileNotFoundError as exc:
            print(f"{pkg} | <error: {exc}>")
            continue
        strategy, note = classify(config)
        base_image = config.get("base_image", "<none>")
        suffix = f" -- {note}" if note else ""
        print(f"{pkg:32s} | {strategy:14s} | {base_image}{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pkg", action="append", default=[], help="Package name (repeatable)")
    parser.add_argument("--all-gpu-packages", action="store_true", help="Every package used by a gpus>0 task")
    parser.add_argument("--only-strategy", choices=[Strategy.PYTORCH, Strategy.NVIDIA_CUDA], help="Filter --all-gpu-packages to one strategy")
    parser.add_argument("--list", action="store_true", help="Print base_image + strategy per package and exit; no build")
    parser.add_argument("--dry-run", action="store_true", help="Extract + write the .def file but skip the apptainer build")
    parser.add_argument("--force", action="store_true", help="Rebuild even if output sif already exists")
    parser.add_argument("--allow-needs-review", action="store_true", help="Attempt needs-review packages too (best-effort, unverified translation)")
    parser.add_argument("--source-cache-dir", type=Path, default=DEFAULT_SOURCE_CACHE_DIR, help="Where to find/pull the existing (glibc-broken) base sif")
    parser.add_argument("--output-cache-dir", type=Path, default=DEFAULT_OUTPUT_CACHE_DIR, help="Where to write rebuilt sifs (never the same as --source-cache-dir)")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR, help="Scratch space for extraction + .def files")
    args = parser.parse_args()

    if args.output_cache_dir.resolve() == args.source_cache_dir.resolve():
        parser.error("--output-cache-dir must differ from --source-cache-dir (never overwrite the shared cache in place)")

    if args.all_gpu_packages:
        pkgs = discover_gpu_packages()
    else:
        pkgs = args.pkg
    if not pkgs:
        parser.error("Pass --pkg NAME (repeatable) or --all-gpu-packages")

    if args.only_strategy:
        filtered = []
        for pkg in pkgs:
            try:
                strategy, _ = classify(load_pkg_config(pkg))
            except FileNotFoundError:
                continue
            if strategy == args.only_strategy:
                filtered.append(pkg)
        pkgs = filtered

    if args.list:
        cmd_list(pkgs)
        return

    for pkg in pkgs:
        try:
            build_one(
                pkg,
                source_cache_dir=args.source_cache_dir,
                output_cache_dir=args.output_cache_dir,
                work_dir=args.work_dir,
                force=args.force,
                dry_run=args.dry_run,
                allow_needs_review=args.allow_needs_review,
            )
        except subprocess.CalledProcessError as exc:
            print(f"[{pkg}] FAILED: {exc}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 -- keep going across a batch
            print(f"[{pkg}] ERROR: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
