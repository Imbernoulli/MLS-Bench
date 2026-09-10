"""The docker variant's Compose overlay: /dev/shm plus the NVIDIA reservation.

Kept free of the renderer's dependencies (Jinja2, tomli_w) on purpose:
``mls_bench.harbor_env`` writes this overlay per trial from inside a plain
``uv tool install harbor`` environment, which has neither.
"""
from __future__ import annotations

# /dev/shm for a task container on local Docker, matching the `--shm-size=16g`
# MLS-Bench's native docker path uses (src/mlsbench/agent/tools.py).
SHM_SIZE_GB = 16


def compose_overlay_text(gpus: int, gpu_ids: list[str] | None = None) -> str:
    """Compose overlay text: ``shm_size`` plus the device reservation when gpus > 0."""
    text = "services:\n  main:\n" f"    shm_size: {SHM_SIZE_GB}gb\n"
    if gpus > 0:
        text += (
            "    deploy:\n"
            "      resources:\n"
            "        reservations:\n"
            "          devices:\n"
            "            - driver: nvidia\n"
        )
        if gpu_ids:
            text += "              device_ids: [" + ", ".join(f'"{g}"' for g in gpu_ids[:gpus]) + "]\n"
        else:
            text += f"              count: {gpus}\n"
        text += "              capabilities: [gpu]\n"
    return text
