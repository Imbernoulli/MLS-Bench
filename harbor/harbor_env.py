"""Local Docker and Daytona environment extensions for MLS-Bench.

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

The ``DockerGPUEnvironment`` remains available for local Docker runs.  The
``DaytonaEnvironment`` implementation is shared with the adapter package and
handles MLS-Bench's GPU-only Compose overlays as direct Daytona GPU sandboxes.

Wired into ``run.yaml`` for remote runs:

    environment:
      import_path: harbor_env:DaytonaEnvironment
"""

from __future__ import annotations

import sys
from pathlib import Path

from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.environments.docker.docker import DockerEnvironment

# ``run-daytona.yaml`` is normally invoked from this directory, so the repo
# root is not automatically on ``sys.path``.  Load the canonical adapter
# implementation without requiring a package installation.
_ADAPTER_SRC = Path(__file__).resolve().parents[1] / "harbor_adapter" / "src"
if str(_ADAPTER_SRC) not in sys.path:
    sys.path.insert(0, str(_ADAPTER_SRC))
from mls_bench.harbor_env import (  # noqa: E402
    DaytonaEnvironment,
    DaytonaClientManager,
    is_gpu_reservation_only_compose,
)


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


__all__ = [
    "DockerGPUEnvironment",
    "DaytonaEnvironment",
    "DaytonaClientManager",
    "is_gpu_reservation_only_compose",
]
