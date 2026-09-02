"""Harbor environment subclass that enables GPU passthrough on vanilla Docker.

Harbor's stock ``DockerEnvironment`` reports ``capabilities.gpus = False``
and rejects any task with ``[environment].gpus > 0`` with the message
"Please use a GPU-capable environment type (e.g., Modal, Docker with
nvidia-docker)".  That guidance is out of date for everyday research
machines: any Linux host with the NVIDIA Container Toolkit installed
(``nvidia-container-runtime`` in ``docker info`` -> ``Runtimes``) can run
GPU containers directly via ``docker compose`` without going through
Modal or another paid backend.

This module exposes a thin subclass that flips that one flag without
modifying Harbor's source.  The MLS-Bench Harbor adapter's per-task
``environment/docker-compose.yaml`` (emitted by ``adapter.py`` when
``gpus > 0``) reserves the actual nvidia devices via the standard

    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: N
              capabilities: [gpu]

block.  Harbor merges that compose file with its base via
``harbor/environments/docker/docker.py::_docker_compose_paths`` so the
nvidia runtime attaches without any extra plumbing.

Use this class through Harbor's standard plugin loader:

    harbor run -p <task-dir> -a oracle \\
        --environment-import mls_bench.harbor_env:DockerGPUEnvironment

or in a Harbor job YAML (``run_mls-bench.yaml``):

    environment:
      import_path: mls_bench.harbor_env:DockerGPUEnvironment

CPU-only tasks (``gpus = 0`` in their ``task.toml``) work identically to
the stock ``DockerEnvironment`` — the GPU capability is simply available
but unused.
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import logging
import sys
import tomllib
from pathlib import Path

from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.environments.docker.docker import DockerEnvironment
from harbor.environments.singularity import singularity as _singularity_mod
from harbor.environments.singularity.singularity import SingularityEnvironment

if sys.platform != "win32":
    import fcntl


# ---------------------------------------------------------------------------
# --nv GPU passthrough for SingularityEnvironment
# ---------------------------------------------------------------------------
# SingularityEnvironment._start_server() builds its `singularity exec` command
# inline (~160 lines, no override seam) and never passes --nv, so GPU tasks
# get a container with no NVIDIA driver/libraries bound in at all. There's no
# clean way to override just that one flag without duplicating the whole
# method — so instead this patches asyncio.create_subprocess_exec itself,
# narrowly scoped to calls that are exactly `singularity exec ...`, and only
# while the currently-running trial actually wants GPUs.
#
# "Currently running trial" is tracked with a contextvar rather than a plain
# module attribute: asyncio.Task creation copies the current context, so a
# value set in SingularityGPUEnvironment.start() before any await correctly
# propagates down through `await self._start_server()` for *that trial's*
# call tree without leaking into a concurrently-running trial for a
# different, non-GPU task in the same `harbor run --n-concurrent` process.
_WANTS_NV: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "mls_bench_wants_nv", default=False
)

_real_create_subprocess_exec = asyncio.create_subprocess_exec
_nv_logger = logging.getLogger(__name__)


async def _nv_aware_create_subprocess_exec(*cmd, **kwargs):
    if (
        _WANTS_NV.get()
        and len(cmd) >= 2
        and cmd[0] == "singularity"
        and cmd[1] == "exec"
        and "--nv" not in cmd
    ):
        cmd = (cmd[0], cmd[1], "--nv", *cmd[2:])
        _nv_logger.debug(f"mls_bench: injected --nv: {cmd}")
    return await _real_create_subprocess_exec(*cmd, **kwargs)


# Patched once at import time. Guarded by identity check so re-importing this
# module (e.g. module reload) doesn't wrap the wrapper.
if _singularity_mod.asyncio.create_subprocess_exec is not _nv_aware_create_subprocess_exec:
    _singularity_mod.asyncio.create_subprocess_exec = _nv_aware_create_subprocess_exec


class DockerGPUEnvironment(DockerEnvironment):
    """``DockerEnvironment`` with ``capabilities.gpus = True``.

    Inherits build, exec, upload, capability validators, etc. from the
    stock environment.  The only difference is that ``_validate_gpu_support``
    (defined on ``BaseEnvironment``) sees ``capabilities.gpus = True`` and
    accepts tasks that declare ``gpus > 0``.
    """

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        base = super().capabilities
        return EnvironmentCapabilities(
            gpus=True,
            disable_internet=base.disable_internet,
            windows=base.windows,
            mounted=base.mounted,
        )


class SingularityGPUEnvironment(SingularityEnvironment):
    """``SingularityEnvironment`` extended to run MLS-Bench's rendered Harbor tasks.

    MLS-Bench's adapter renders a thin per-task ``environment/Dockerfile``
    (``FROM <harbor-base>`` + ``COPY _scaffold/ <workdir>/``) rather than
    setting ``[environment].docker_image`` directly, so that the task's
    editable-surface scaffold gets layered onto the shared per-package base
    image at build time (see ``harbor_adapter/README.md``). Stock
    ``SingularityEnvironment`` has no build step at all — it only pulls a
    pre-built ``docker_image`` — so it can never apply that scaffold layer,
    and it also rejects any task with ``gpus > 0`` outright (capabilities.gpus
    defaults to False). This subclass adds:

    1. A Dockerfile-aware build step: parses the ``FROM``/``COPY`` lines
       (only these two are supported, matching the fixed template the
       adapter emits) and builds a derived per-task ``.sif`` via
       ``apptainer build`` from a generated definition file —
       ``Bootstrap: localimage`` on the cached base ``.sif`` (itself pulled
       through the stock ``_convert_docker_to_sif``, so it's shared across
       every task built from the same package) plus a ``%files`` section.
       Docker's ``COPY <src>/ <dst>/`` recursively *merges* src's contents
       into dst; apptainer's ``%files`` uses plain ``cp -r`` semantics, which
       *nests* a directory entry under an already-existing same-named dst
       (true here — the base image already has ``<workdir>/<pkg>/``
       populated). So every scaffold file is expanded to its own ``%files``
       line with its exact destination path, however deep the nesting —
       never a directory-level entry.
    2. GPU passthrough: ``capabilities.gpus = True`` (mirrors
       ``DockerGPUEnvironment`` above, unblocking ``_validate_gpu_support``)
       *and* an actual ``--nv`` flag on ``singularity exec`` for any task
       with ``gpus > 0`` — see the ``_WANTS_NV`` / monkeypatch block near the
       top of this module for how, since ``_start_server`` builds that
       command inline with no override seam to hook into directly.
    3. A ``workdir`` fix: the stock class defaults to ``/app`` whenever no
       ``WORKDIR`` line is present (true for every rendered task — the
       template never emits one). Ours reads the ``COPY`` line's
       destination instead, which is exactly the package's workdir
       (``/workspace`` by default, per ``build_base_image.py``).
    4. A generous default ``exec()`` timeout: the stock class's HTTP client
       hardcodes a 600s timeout whenever a caller doesn't pass
       ``timeout_sec`` explicitly (``singularity.py``'s
       ``_DEFAULT_HTTP_TIMEOUT``). Harbor's shared-mode verifier is exactly
       such a caller — it wraps the same exec() call in its own
       ``asyncio.wait_for(..., timeout=<task's [verifier].timeout_sec>)``
       (up to ~67000s across MLS-Bench tasks, 18000s for the vast
       majority), so the real per-task budget is already enforced one layer
       up. Without this override, any verifier run whose eval scripts take
       longer than 10 minutes — easily the case for an agent-submitted
       optimizer that's slower per step than the baseline — gets silently
       truncated at 600s regardless of what the task allows (confirmed by
       hand: a GPU trial's verifier phase died with ``HTTP request timed
       out after 600 seconds`` while the task declared a 5700s budget).

    Per-task ``.sif`` builds are cached in the same
    ``singularity_image_cache_dir`` as the base-image pulls, keyed by a
    content hash of the scaffold so edited scaffolds rebuild automatically
    while unchanged ones reuse the cached artifact.

    Use via Harbor's plugin loader:

        harbor run -p <task-dir> \\
            --environment-import-path mls_bench.harbor_env:SingularityGPUEnvironment

    or in a Harbor job YAML:

        environment:
          import_path: mls_bench.harbor_env:SingularityGPUEnvironment
    """

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        base = super().capabilities
        return EnvironmentCapabilities(
            gpus=True,
            disable_internet=base.disable_internet,
            windows=base.windows,
            mounted=base.mounted,
        )

    # Fallback only -- used when task.toml is missing/unparseable/lacks
    # [verifier].timeout_sec. Comfortably above the largest observed
    # MLS-Bench value (~67000s) so it never becomes the effective bottleneck.
    _DEFAULT_EXEC_TIMEOUT_SEC = 86400

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        if timeout_sec is None:
            timeout_sec = self._task_verifier_timeout_sec()
        return await super().exec(
            command=command, cwd=cwd, env=env, timeout_sec=timeout_sec, user=user
        )

    def _task_verifier_timeout_sec(self) -> int:
        """Read ``[verifier].timeout_sec`` straight from the task's ``task.toml``.

        The environment object is only constructed with ``[environment]``
        config (``task_env_config``), not ``[verifier]``, so this reads the
        file directly rather than guessing a value -- ``environment_dir`` is
        always ``<task_dir>/environment`` (see ``_dockerfile_path`` on the
        base class), so ``environment_dir.parent / "task.toml"`` is the
        task's own definition.
        """
        task_toml = self.environment_dir.parent / "task.toml"
        try:
            with open(task_toml, "rb") as f:
                data = tomllib.load(f)
            return int(data["verifier"]["timeout_sec"])
        except (OSError, KeyError, ValueError, tomllib.TOMLDecodeError):
            return self._DEFAULT_EXEC_TIMEOUT_SEC

    @property
    def _uses_dockerfile_build(self) -> bool:
        """True when this task has no docker_image and must be built."""
        return not self._docker_image and self._dockerfile_path.exists()

    def _validate_definition(self) -> None:
        if self._uses_dockerfile_build:
            self._parse_dockerfile()  # raises on anything unsupported
            return
        super()._validate_definition()

    def _parse_dockerfile(self) -> tuple[str, list[tuple[str, str]]]:
        """Parse the per-task Dockerfile's FROM + COPY lines.

        Only ``FROM <image>`` and ``COPY <src> <dst>`` are supported — this
        matches the fixed template MLS-Bench's adapter emits
        (``harbor_adapter/.../environment/Dockerfile.j2``); anything else
        raises rather than silently doing the wrong thing.

        Returns (base_image, [(src, dst), ...]) with src/dst as written in
        the Dockerfile (src relative to environment_dir).
        """
        base_image: str | None = None
        copies: list[tuple[str, str]] = []
        for line in self._dockerfile_path.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            upper = stripped.upper()
            if upper.startswith("FROM "):
                base_image = stripped.split(None, 1)[1].strip()
            elif upper.startswith("COPY "):
                parts = stripped.split()
                if len(parts) != 3:
                    raise ValueError(
                        f"Unsupported COPY line in {self._dockerfile_path} "
                        f"(only 'COPY <src> <dst>' is supported): {stripped!r}"
                    )
                copies.append((parts[1], parts[2]))
            else:
                raise ValueError(
                    f"Unsupported Dockerfile instruction in {self._dockerfile_path} "
                    f"(only FROM/COPY are supported): {stripped!r}"
                )
        if not base_image:
            raise ValueError(f"No FROM line found in {self._dockerfile_path}")
        return base_image, copies

    def _resolve_workdir(self) -> str:
        if self._uses_dockerfile_build:
            try:
                _, copies = self._parse_dockerfile()
            except ValueError:
                return "/workspace"
            if copies:
                return copies[-1][1].rstrip("/") or "/"
            return "/workspace"
        return super()._resolve_workdir()

    def _scaffold_entries(self, src: str, dst: str) -> list[tuple[Path, str]]:
        """Expand one Docker COPY line into literal (host_path, container_dest) pairs.

        Docker's ``COPY <src>/ <dst>/`` recursively *merges* src's contents
        into dst, overwriting individual files but never nesting src under
        an existing same-named dst directory. apptainer's ``%files`` copies
        each entry with plain ``cp -r`` semantics: if the destination
        directory already exists (true here — the base image already has
        ``<workdir>/<pkg>/`` populated with the package source), a
        directory-level entry gets nested *under* it instead of merged.
        So every entry must be a single leaf **file**, individually mapped
        to its exact destination path, however deep the scaffold nesting.
        """
        host_src = self.environment_dir / src
        dst_clean = dst.rstrip("/")
        if host_src.is_file():
            return [(host_src, f"{dst_clean}/{host_src.name}")]
        if not host_src.is_dir():
            raise FileNotFoundError(f"COPY source not found: {host_src}")
        return [
            (p, f"{dst_clean}/{p.relative_to(host_src).as_posix()}")
            for p in sorted(host_src.rglob("*"))
            if p.is_file()
        ]

    def _scaffold_hash(self, copies: list[tuple[str, str]]) -> str:
        """Content hash of every file a COPY line would place, for cache keying."""
        h = hashlib.sha256()
        for src, dst in copies:
            for host_path, container_dest in self._scaffold_entries(src, dst):
                h.update(container_dest.encode())
                h.update(host_path.read_bytes())
        return h.hexdigest()[:16]

    async def _build_task_sif(self, *, force_build: bool) -> Path:
        """Build (or reuse a cached) per-task .sif with the scaffold applied."""
        base_image, copies = self._parse_dockerfile()
        base_sif = await self._convert_docker_to_sif(base_image, force_pull=force_build)

        if not copies:
            # No scaffold to layer on — the base image already is the task image.
            return base_sif

        scaffold_hash = self._scaffold_hash(copies)
        safe_name = self.environment_name.replace("/", "_")
        task_sif_path = self._image_cache_dir / f"task_{safe_name}_{scaffold_hash}.sif"
        lock_path = task_sif_path.with_suffix(".sif.lock")

        if not force_build and task_sif_path.exists():
            self.logger.debug(f"Using cached per-task Singularity image: {task_sif_path}")
            return task_sif_path

        self._image_cache_dir.mkdir(parents=True, exist_ok=True)
        lock_file = open(lock_path, "w")
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, lambda: fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            )

            if task_sif_path.exists() and not force_build:
                self.logger.debug(
                    f"Using per-task Singularity image built by another process: "
                    f"{task_sif_path}"
                )
                return task_sif_path

            def_lines = ["Bootstrap: localimage", f"From: {base_sif}", "", "%files"]
            for src, dst in copies:
                for host_path, container_dest in self._scaffold_entries(src, dst):
                    def_lines.append(f"{host_path} {container_dest}")

            def_path = self._image_cache_dir / f"task_{safe_name}_{scaffold_hash}.def"
            def_path.write_text("\n".join(def_lines) + "\n")

            tmp_sif = (
                self._image_cache_dir
                / f"task_{safe_name}_{scaffold_hash}.sif.tmp.{self.session_id}"
            )
            cmd = ["apptainer", "build", "--force", str(tmp_sif), str(def_path)]
            self.logger.debug(f"Building per-task Singularity image: {' '.join(cmd)}")

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                if tmp_sif.exists():
                    tmp_sif.unlink()
                raise RuntimeError(
                    "Failed to build per-task Singularity image:\n"
                    f"{stderr.decode(errors='replace')}"
                )

            tmp_sif.rename(task_sif_path)
            self.logger.debug(f"Built per-task Singularity image: {task_sif_path}")
            return task_sif_path
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()

    async def start(self, force_build: bool) -> None:
        # See the _WANTS_NV / _nv_aware_create_subprocess_exec block near the
        # top of this module: this is what actually turns on --nv for this
        # trial's `singularity exec` call, scoped so a concurrently-running
        # CPU-only trial in the same process isn't affected.
        token = _WANTS_NV.set(self._effective_gpus > 0)
        try:
            if not self._uses_dockerfile_build:
                await super().start(force_build)
                return
            if sys.platform == "win32":
                raise RuntimeError(
                    "SingularityGPUEnvironment is only supported on Linux/macOS."
                )
            self._sif_path = await self._build_task_sif(force_build=force_build)
            await self._start_server()
            await self._upload_environment_dir_after_start()
        finally:
            _WANTS_NV.reset(token)
