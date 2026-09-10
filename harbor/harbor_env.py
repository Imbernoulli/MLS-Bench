"""Local Docker and Daytona environment extensions for MLS-Bench.

Harbor's stock `type: docker` environment reports `capabilities.gpus = False`
and rejects any task with `[environment].gpus > 0` ("Please use a GPU-capable
environment type"). That guidance is out of date for hosts with the NVIDIA
Container Toolkit installed (`nvidia-container-runtime` in `docker info` →
Runtimes), which can run GPU containers directly via `docker compose`.

``DockerGPUEnvironment`` flips the one flag without modifying Harbor's
source and writes a per-trial compose overlay with the NVIDIA device
reservation (``--ek gpu_ids=2,3`` pins devices) and ``shm_size: 16gb``,
ignoring the bundle's own overlay so devices are never reserved twice. A CPU
request above the host's core count is clamped with a warning.
``--ek gpu_type=H200`` selects a task's native H200 profile for the agent and
the verifier alike, as it does on Daytona.

Both classes live in the adapter package (``mls_bench.harbor_env``) and are
re-exported here so ``run.yaml`` / ``run-daytona.yaml`` can name them as
``harbor_env:...`` from this directory.  ``DaytonaEnvironment`` is optional:
``tasks-daytona/`` runs on stock Harbor's ``daytona`` environment as shipped;
this class adds the H100 default, clamp-on-refusal, toolbox retries and the
H200 profile export.  ``ModalEnvironment`` likewise: ``tasks-modal/`` runs on
stock Harbor's ``modal`` environment; the class only adds ``--ek
gpu_type=H200``.

Wired into ``run.yaml`` for remote runs:

    environment:
      import_path: harbor_env:DaytonaEnvironment
"""

from __future__ import annotations

import sys
from pathlib import Path

# ``run-daytona.yaml`` is normally invoked from this directory, so the repo
# root is not automatically on ``sys.path``.  Load the canonical adapter
# implementation without requiring a package installation.
_ADAPTER_SRC = Path(__file__).resolve().parents[1] / "harbor_adapter" / "src"
if str(_ADAPTER_SRC) not in sys.path:
    sys.path.insert(0, str(_ADAPTER_SRC))
from mls_bench.harbor_env import (  # noqa: E402
    DaytonaEnvironment,
    DaytonaClientManager,
    DockerGPUEnvironment,
    ModalEnvironment,
    is_gpu_reservation_only_compose,
)


__all__ = [
    "DockerGPUEnvironment",
    "DaytonaEnvironment",
    "DaytonaClientManager",
    "ModalEnvironment",
    "is_gpu_reservation_only_compose",
]
