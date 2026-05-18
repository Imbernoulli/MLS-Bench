"""DockerGPUEnvironment — flip Harbor's stock DockerEnvironment GPU flag.

Harbor's stock `type: docker` environment reports `capabilities.gpus = False`
and rejects any task with `[environment].gpus > 0` ("Please use a GPU-capable
environment type"). That guidance is out of date for hosts with the NVIDIA
Container Toolkit installed (`nvidia-container-runtime` in `docker info` →
Runtimes), which can run GPU containers directly via `docker compose`.

This subclass flips the one flag without modifying Harbor's source. Each
MLS-Bench task that needs GPUs ships an `environment/docker-compose.yaml`
that reserves nvidia devices via the standard
`deploy.resources.reservations.devices` block; Harbor merges that with its
base compose file. CPU-only tasks (`gpus = 0`) work identically to the stock
environment — the GPU capability is just declared, not used.

Wired into `run.yaml`:

    environment:
      import_path: harbor_env:DockerGPUEnvironment
"""

from __future__ import annotations

from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.environments.docker.docker import DockerEnvironment


class DockerGPUEnvironment(DockerEnvironment):
    @property
    def capabilities(self) -> EnvironmentCapabilities:
        base = super().capabilities
        return EnvironmentCapabilities(
            gpus=True,
            disable_internet=base.disable_internet,
            windows=base.windows,
            mounted=base.mounted,
        )
