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

from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.environments.docker.docker import DockerEnvironment


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
